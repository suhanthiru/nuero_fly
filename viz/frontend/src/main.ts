/**
 * The 3D brain view, driven by simulated activity.
 *
 * Design rules this file is required to honour:
 *   - colour encodes cell type, brightness encodes activity; activity never touches hue
 *   - emissive brightness only, no bloom and no depth of field, so the image stays
 *     quantitative and comparable to calcium imaging
 *   - the escape pathway is the subject; everything else is dim anatomical context
 *
 * deck.gl rather than a scene graph, because the thing that changes every frame is a
 * per-object scalar in an attribute buffer, not the structure of a scene.
 *
 * Layers are split into static (shells, context cloud) and activity-driven (pathway somata,
 * skeleton bundles, identified-cell meshes). Only the latter are rebuilt, and only when a
 * new frame arrives - rebuilding everything per camera frame is what made orbiting unusable
 * before.
 */

import { Deck, OrbitView, type PickingInfo } from '@deck.gl/core';
import { LineLayer, PointCloudLayer } from '@deck.gl/layers';
import { SimpleMeshLayer } from '@deck.gl/mesh-layers';

import { ArenaPanel } from './arena';
import { loadScene, partition, type Scene } from './scene';
import { ActivityIndex, Stream, type Frame, type Hello } from './stream';
import { TracePanel } from './traces';

const STREAM_URL = `ws://${location.hostname}:8000/stream`;

const INITIAL_VIEW = {
  target: [0, 0, 0] as [number, number, number],
  rotationX: 12,
  rotationOrbit: 24,
  zoom: 0.08,
  minZoom: -3,
  maxZoom: 6,
};

// Draw only back faces - the far wall of the hull. The near wall would sit between the
// camera and the neurons, veiling the subject; the far wall instead becomes a backdrop the
// neurons read against.
const SHELL_PARAMETERS = {
  depthWriteEnabled: false,
  cullMode: 'front',
  blend: true,
  blendColorOperation: 'add',
  blendColorSrcFactor: 'src-alpha',
  blendColorDstFactor: 'one',
  blendAlphaOperation: 'add',
  blendAlphaSrcFactor: 'one',
  blendAlphaDstFactor: 'one',
} as const;

// Activity spans more than two orders of magnitude across the pathway - the LC populations
// sit far below the giant fiber - so a linear ramp would render the bundles black. This
// gamma is a display transform only; the streamed values are untouched, and the subtitle
// states both it and what full brightness corresponds to.
const DISPLAY_GAMMA = 0.45;

// Below this, a neuron is drawn at its resting colour rather than lit, so the baseline does
// not read as activity.
const FLOOR = 0.02;

type ViewState = typeof INITIAL_VIEW;

let scene: Scene;
let deck: Deck<any>;
let viewState: ViewState = { ...INITIAL_VIEW };
let showContext = true;
let showShells = true;
let showMorphology = true;

let parts: ReturnType<typeof partition>;
let staticLayers: any[] | null = null;
let index: ActivityIndex | null = null;
let lastStep = -1;

const stream = new Stream(STREAM_URL);
const traces = new TracePanel(document.getElementById('traces') as HTMLCanvasElement);
const arena = new ArenaPanel(document.getElementById('arena-canvas') as HTMLCanvasElement);

/** Per-neuron colour buffers, allocated once and mutated in place each frame. */
let pathwayColour: Uint8Array;
let pathwayBase: Uint8Array;

function centroid(positions: Float32Array): [number, number, number] {
  let x = 0, y = 0, z = 0;
  const n = positions.length / 3;
  for (let i = 0; i < n; i++) {
    x += positions[i * 3];
    y += positions[i * 3 + 1];
    z += positions[i * 3 + 2];
  }
  return [x / n, y / n, z / n];
}

function halfCentroid(points: Float32Array, xSign: number): [number, number, number] {
  let x = 0, y = 0, z = 0, n = 0;
  for (let i = 0; i < points.length / 3; i++) {
    if (Math.sign(points[i * 3]) !== xSign) continue;
    x += points[i * 3];
    y += points[i * 3 + 1];
    z += points[i * 3 + 2];
    n++;
  }
  return n ? [x / n, y / n, z / n] : [0, 0, 0];
}

/** Streamed 0..255 -> display brightness 0..1. */
function brightness(raw: number): number {
  const value = raw / 255;
  return value < FLOOR ? 0 : Math.pow(value, DISPLAY_GAMMA);
}

// --- layers ----------------------------------------------------------------------

function buildStaticLayers(): any[] {
  const layers: any[] = [];

  if (showShells) {
    for (const shell of scene.shells) {
      layers.push(
        new SimpleMeshLayer({
          id: `shell-${shell.name}`,
          data: [{ position: [0, 0, 0] }],
          mesh: {
            attributes: { POSITION: { value: shell.positions, size: 3 } },
            indices: { value: shell.indices, size: 1 },
          },
          getPosition: (d: any) => d.position,
          getColor: [...shell.color, Math.round(shell.opacity * 255)],
          material: false,
          pickable: false,
          parameters: SHELL_PARAMETERS,
        }),
      );
    }
  }

  if (showContext) {
    layers.push(
      new PointCloudLayer({
        id: 'somata-context',
        data: {
          length: parts.context.count,
          attributes: {
            getPosition: { value: parts.context.position, size: 3 },
            getColor: { value: parts.context.color, size: 3 },
          },
        },
        pointSize: 1.0,
        // Standard alpha blending, not additive: ~90k of the 140k somata sit in the optic
        // lobe rind, and additive accumulation there saturates to white and beats the
        // pathway it is meant to sit behind.
        opacity: showMorphology ? 0.22 : 0.62,
        material: false,
        pickable: false,
        parameters: { depthWriteEnabled: false },
      }),
    );
  }
  return layers;
}

function buildActivityLayers(frame: Frame | null): any[] {
  const layers: any[] = [];
  const aggregate = frame?.aggregate ?? {};
  const scale = stream.hello?.meta?.display_scale ?? 1;

  const typeBrightness = (type: string): number => {
    const raw = (aggregate[type] ?? 0) / (scale || 1);
    return raw < FLOOR ? 0 : Math.pow(Math.min(raw, 1), DISPLAY_GAMMA);
  };

  if (showMorphology) {
    for (const bundle of scene.bundles) {
      // Populations are lit by their cell-type mean. Per-axon brightness would need a body
      // id per skeleton, which the geometry export does not currently carry.
      const lit = typeBrightness(bundle.name);
      const colour = bundle.color.map((c) => Math.round(c * (0.25 + 0.75 * lit)));
      layers.push(
        new LineLayer({
          id: `bundle-${bundle.name}`,
          data: {
            length: bundle.count,
            attributes: {
              getSourcePosition: { value: bundle.source, size: 3 },
              getTargetPosition: { value: bundle.target, size: 3 },
            },
          },
          getColor: colour as unknown as [number, number, number],
          getWidth: 1,
          widthUnits: 'pixels',
          widthMinPixels: 0.7,
          // Low, because these bundles are dense by construction and additive segments
          // overlapping in the glomerulus saturate to white at any higher value.
          opacity: 0.1 + 0.16 * lit,
          pickable: false,
          updateTriggers: { getColor: colour.join(',') },
          parameters: {
            depthWriteEnabled: false,
            blend: true,
            blendColorOperation: 'add',
            blendColorSrcFactor: 'src-alpha',
            blendColorDstFactor: 'one',
            blendAlphaOperation: 'add',
            blendAlphaSrcFactor: 'one',
            blendAlphaDstFactor: 'one',
          },
        }),
      );
    }

    for (const neuron of scene.neuronMeshes) {
      let lit = typeBrightness(neuron.name);
      const column = stream.hello?.body_ids.indexOf(String(neuron.bodyId)) ?? -1;
      if (frame && column >= 0) lit = brightness(frame.activity[column]);
      const colour = neuron.color.map((c) => Math.round(c * (0.28 + 0.72 * lit)));
      layers.push(
        new SimpleMeshLayer({
          id: `neuron-${neuron.name}-${neuron.bodyId}`,
          data: [{ position: [0, 0, 0] }],
          mesh: {
            attributes: { POSITION: { value: neuron.positions, size: 3 } },
            indices: { value: neuron.indices, size: 1 },
          },
          getPosition: (d: any) => d.position,
          getColor: [...colour, 235] as unknown as [number, number, number, number],
          material: false,
          pickable: false,
          updateTriggers: { getColor: colour.join(',') },
          // Closed surfaces, so culling back faces halves fragment work for free.
          parameters: { depthWriteEnabled: true, cullMode: 'back' },
        }),
      );
    }
  }

  // Pathway somata: per-neuron brightness, the finest-grained thing on screen.
  if (frame && index) {
    for (let i = 0; i < parts.pathway.count; i++) {
      const column = index.columnForSoma[i];
      const lit = column >= 0 ? brightness(frame.activity[column]) : 0;
      const gain = 0.22 + 0.78 * lit;
      pathwayColour[i * 3] = pathwayBase[i * 3] * gain;
      pathwayColour[i * 3 + 1] = pathwayBase[i * 3 + 1] * gain;
      pathwayColour[i * 3 + 2] = pathwayBase[i * 3 + 2] * gain;
    }
  }

  layers.push(
    new PointCloudLayer({
      id: 'somata-pathway',
      data: {
        length: parts.pathway.count,
        attributes: {
          getPosition: { value: parts.pathway.position, size: 3 },
          getColor: { value: pathwayColour, size: 3 },
        },
      },
      pointSize: 5.5,
      opacity: 1,
      material: false,
      pickable: true,
      updateTriggers: { getColor: lastStep },
      // Drawn last and depth-tested against nothing, so the subject is never lost inside
      // the cloud.
      parameters: { depthWriteEnabled: false, depthCompare: 'always' },
      onHover: (info: PickingInfo) => showHover(info),
    }),
  );

  return layers;
}

function redraw(): void {
  staticLayers ??= buildStaticLayers();
  deck.setProps({
    layers: [...staticLayers, ...buildActivityLayers(stream.frame)],
    viewState: viewState as any,
  });
}

function invalidateStatic(): void {
  staticLayers = null;
  redraw();
}

/** Camera-only update: deliberately does not rebuild layers. */
function updateCamera(): void {
  deck.setProps({ viewState: viewState as any });
}

function flyTo(next: Partial<ViewState>): void {
  viewState = { ...viewState, ...next };
  updateCamera();
}

function showHover(info: PickingInfo): void {
  const element = document.getElementById('hover')!;
  if (info.index < 0 || !info.picked) {
    element.style.display = 'none';
    return;
  }
  const i = parts.pathway.index[info.index];
  const type = scene.somata.cellTypes[scene.somata.typeIndex[i]];
  const column = index ? index.columnForSoma[info.index] : -1;
  const value =
    stream.frame && column >= 0
      ? `${(stream.frame.activity[column] / 255).toFixed(2)} rel.`
      : 'no signal';
  element.innerHTML =
    `${type}<br><span class="s">body ${scene.somata.bodyId[i]} &middot; ${value}</span>`;
  element.style.display = 'block';
  element.style.left = `${info.x + 14}px`;
  element.style.top = `${info.y + 14}px`;
}

// --- panels and HUD ---------------------------------------------------------------

const text = (id: string, value: string) => {
  const node = document.getElementById(id);
  if (node) node.textContent = value;
};

function configureTraces(hello: Hello): void {
  const history = hello.history;
  const scale = hello.meta?.display_scale ?? 1;
  const normalise = (series: number[]) => series.map((v) => v / (scale || 1));
  const peak = (series: number[] | undefined, floor: number) =>
    Math.max(floor, ...(series && series.length ? series : [floor]));

  const rows = [
    { key: 'θ', colour: '#94a3b8',
      scale: peak(history.theta_deg, 10), series: history.theta_deg ?? [] },
    { key: 'θ̇', colour: '#cbd5e1',
      scale: peak(history.theta_dot, 0.01), series: history.theta_dot ?? [] },
  ];
  for (const type of ['LC4', 'LPLC2', 'DNp01', 'GFC2', 'TTMn']) {
    const series = history.aggregate[type];
    if (!series) continue;
    rows.push({
      key: type,
      colour: hello.palette[type] ?? '#64748b',
      scale: 1,
      series: normalise(series),
    });
  }
  traces.configure(rows, history.stride);

  // Mark the giant fiber spike and the moment of contact on the shared time axis.
  const gfMs = hello.events?.gf_spike_ms ?? null;
  if (gfMs !== null) traces.mark(Math.round(gfMs / hello.dt_ms), 'GF');
  if (hello.collision_ms !== null) {
    traces.mark(Math.round(hello.collision_ms / hello.dt_ms), 'contact');
  }
}

function updatePanels(frame: Frame, hello: Hello): void {
  const gfMs = hello.events?.gf_spike_ms ?? null;
  const fired = gfMs !== null && frame.t_ms >= gfMs;

  arena.draw({
    thetaDeg: frame.theta_deg ?? 0,
    thetaDot: frame.theta_dot ?? 0,
    distanceMm: frame.distance_mm ?? 0,
    startDistanceMm: startDistance,
    tMs: frame.t_ms,
    collisionMs: hello.collision_ms,
    fired,
  });

  text('hud-t', `${frame.t_ms.toFixed(1)} ms`);
  text('hud-theta', `${(frame.theta_deg ?? 0).toFixed(1)}°`);
  text('hud-thetadot', `${(frame.theta_dot ?? 0).toFixed(3)} °/ms`);
  text('hud-ratio', hello.ratio_ms === null ? '—' : `${hello.ratio_ms.toFixed(0)} ms`);
  text(
    'hud-contact',
    hello.collision_ms === null
      ? '—'
      : `${(hello.collision_ms - frame.t_ms).toFixed(0)} ms`,
  );

  const gfCell = document.getElementById('hud-gf')!;
  gfCell.textContent = gfMs === null ? 'silent' : `${gfMs.toFixed(1)} ms`;
  gfCell.className = fired ? 'fired' : '';
  const ttmMs = hello.events?.ttm_spike_ms ?? null;
  text('hud-ttm', ttmMs === null ? 'silent' : `${ttmMs.toFixed(1)} ms`);

  updateLegend(frame, hello);
}

/** Largest distance seen this trial, so the top-down inset has a stable scale. */
let startDistance = 1;

function updateLegend(frame: Frame, hello: Hello): void {
  const scale = hello.meta?.display_scale ?? 1;
  const container = document.getElementById('legend')!;
  const types = Object.keys(frame.aggregate).sort();
  if (container.childElementCount !== types.length) {
    container.innerHTML = types
      .map((type) => {
        const colour = hello.palette[type] ?? '#64748b';
        return (
          `<div class="row" data-type="${type}">` +
          `<span class="sw" style="background:${colour}"></span>${type}` +
          `<span class="bar"><span style="background:${colour};width:0%"></span></span></div>`
        );
      })
      .join('');
  }
  for (const type of types) {
    const bar = container.querySelector(`[data-type="${type}"] .bar span`) as HTMLElement;
    if (!bar) continue;
    const value = Math.min(1, (frame.aggregate[type] ?? 0) / (scale || 1));
    bar.style.width = `${(value * 100).toFixed(1)}%`;
  }
}

// --- controls ---------------------------------------------------------------------

function wireControls(): void {
  const playpause = document.getElementById('playpause') as HTMLButtonElement;
  let playing = true;
  playpause.onclick = () => {
    playing = !playing;
    playpause.textContent = playing ? 'pause' : 'play';
    if (playing) stream.play();
    else stream.pause();
  };

  const speed = document.getElementById('speed') as HTMLInputElement;
  const applySpeed = () => {
    const value = Math.pow(10, Number(speed.value));
    stream.speed(value);
    const label = `1/${Math.round(1 / value)}×`;
    text('speed-label', label);
    text('hud-speed', label);
  };
  speed.value = String(Math.log10(0.02));
  speed.oninput = applySpeed;

  document.getElementById('rerun')!.addEventListener('click', () => {
    const ratio = Number((document.getElementById('ratio') as HTMLInputElement).value);
    const gain = Number((document.getElementById('gain') as HTMLInputElement).value);
    stream.rerun(ratio, gain, 0);
    text('status', 'simulating…');
  });

  window.addEventListener('keydown', (event) => {
    if ((event.target as HTMLElement)?.tagName === 'INPUT') return;
    const views = buildViews();
    const byKey: Record<string, string> = {
      '1': 'anterior', '2': 'lateral', '3': 'dorsal', '4': 'neck', '5': 'glomerulus',
    };
    if (byKey[event.key]) {
      flyTo(views[byKey[event.key]]);
      return;
    }
    switch (event.key) {
      case ' ':
        event.preventDefault();
        playpause.click();
        break;
      case 'm': showMorphology = !showMorphology; invalidateStatic(); break;
      case 'c': showContext = !showContext; invalidateStatic(); break;
      case 's': showShells = !showShells; invalidateStatic(); break;
    }
  });

  applySpeed();
}

function buildViews(): Record<string, Partial<ViewState>> {
  const cv = scene.shells.find((s) => s.name === 'CV');
  const neck = cv ? centroid(cv.positions) : ([0, 0, 0] as [number, number, number]);
  const lc4 = scene.bundles.find((b) => b.name === 'LC4');
  const glomerulus = lc4
    ? halfCentroid(lc4.target, -1)
    : ([0, 0, 0] as [number, number, number]);
  return {
    anterior: { target: [0, 0, 0], rotationX: 0, rotationOrbit: 0, zoom: 0.08 },
    lateral: { target: [0, 0, 0], rotationX: 0, rotationOrbit: 90, zoom: 0.08 },
    dorsal: { target: [0, 0, 0], rotationX: 88, rotationOrbit: 0, zoom: 0.08 },
    neck: { target: neck, rotationX: 6, rotationOrbit: 28, zoom: 1.5 },
    glomerulus: { target: glomerulus, rotationX: 12, rotationOrbit: -55, zoom: 2.2 },
  };
}

// ----------------------------------------------------------------------------------

async function main(): Promise<void> {
  scene = await loadScene();
  parts = partition(scene.somata);
  pathwayBase = new Uint8Array(parts.pathway.color);
  pathwayColour = new Uint8Array(parts.pathway.color.length);
  pathwayColour.set(parts.pathway.color);

  document.getElementById('loading')!.remove();
  text(
    'subtitle',
    `${scene.manifest.dataset} · ${scene.somata.count.toLocaleString()} somata`,
  );

  deck = new Deck({
    canvas: 'canvas',
    views: new OrbitView({ orbitAxis: 'Z', fovy: 50 }),
    initialViewState: viewState as any,
    // Without these the canvas keeps its 300x150 default backing store.
    width: '100%',
    height: '100%',
    useDevicePixels: true,
    controller: { inertia: 250 },
    onViewStateChange: ({ viewState: next }: any) => {
      viewState = next;
      updateCamera();
    },
    parameters: { clearColor: [0.031, 0.035, 0.043, 1] } as any,
    layers: [],
  });

  wireControls();
  redraw();

  // Deep-linking: ?t=360 seeks to that simulation millisecond on connect, ?view=neck
  // picks a camera preset. Both exist so a specific moment can be captured or shared
  // rather than described.
  const params = new URLSearchParams(location.search);

  stream.whenReady((hello) => {
    index = new ActivityIndex(hello, scene.somata.bodyId, parts.pathway.index);
    configureTraces(hello);
    lastStep = -1;
    startDistance = 1;
    text('status', '');

    const seekTo = params.get('t');
    if (seekTo !== null) {
      const step = Math.round(Number(seekTo) / hello.dt_ms);
      stream.seek(step);
      if (params.get('pause') !== '0') stream.pause();
    }
    const view = params.get('view');
    if (view) {
      const views = buildViews();
      if (views[view]) flyTo(views[view]);
    }

    const full = hello.meta?.display_full_scale_hz;
    text(
      'subtitle',
      `${scene.manifest.dataset} · ${scene.somata.count.toLocaleString()} somata · ` +
        `full scale ≈ ${full ? full.toFixed(1) : '?'} Hz · display γ ${DISPLAY_GAMMA}`,
    );
  });
  stream.connect();

  const tick = () => {
    const frame = stream.frame;
    const hello = stream.hello;
    if (frame && hello && frame.step !== lastStep) {
      lastStep = frame.step;
      startDistance = Math.max(startDistance, frame.distance_mm ?? 1);
      updatePanels(frame, hello);
      redraw();
    }
    traces.draw(lastStep < 0 ? 0 : lastStep);
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

main();
