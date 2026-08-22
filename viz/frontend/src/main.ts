/**
 * The 3D brain view.
 *
 * Design rules this file is required to honour:
 *   - colour encodes cell type, brightness encodes activity; activity never touches hue
 *   - emissive brightness only, no bloom and no depth of field, so the image stays
 *     quantitative and comparable to calcium imaging
 *   - the escape pathway is the subject; everything else is dim anatomical context
 *
 * deck.gl rather than a scene graph, because the thing that changes every frame is a
 * per-object scalar in an attribute buffer, not the structure of a scene.
 */

import { Deck, OrbitView, type PickingInfo } from '@deck.gl/core';
import { PointCloudLayer } from '@deck.gl/layers';
import { SimpleMeshLayer } from '@deck.gl/mesh-layers';

import { loadScene, partition, type Scene } from './scene';

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
// neurons read against. Drawing both walls also doubles the accumulated alpha, which is
// what made the shells brighter than the cells they are meant to sit behind.
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

type ViewState = typeof INITIAL_VIEW;

let scene: Scene;
let viewState: ViewState = { ...INITIAL_VIEW };
let showContext = true;
let showShells = true;
let deck: Deck<any>;

function centroid(positions: Float32Array): [number, number, number] {
  let x = 0;
  let y = 0;
  let z = 0;
  const n = positions.length / 3;
  for (let i = 0; i < n; i++) {
    x += positions[i * 3];
    y += positions[i * 3 + 1];
    z += positions[i * 3 + 2];
  }
  return [x / n, y / n, z / n];
}

function buildLayers() {
  const parts = partition(scene.somata);
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
        // Standard alpha blending, not additive. The optic lobe cortical rind holds ~90k
        // of the 140k somata, and additive accumulation there saturates to white at any
        // usable alpha - which inverts the whole hierarchy by making context brighter than
        // the pathway. Alpha blending caps at the source colour, so the rind stays slate
        // and dense regions read as solid rather than incandescent.
        opacity: 0.62,
        material: false,
        pickable: false,
        parameters: { depthWriteEnabled: false },
      }),
    );
  }

  // Drawn last and depth-tested against nothing, so the pathway is never lost inside the
  // cloud. This is the subject of the image; it is allowed to win.
  layers.push(
    new PointCloudLayer({
      id: 'somata-pathway',
      data: {
        length: parts.pathway.count,
        attributes: {
          getPosition: { value: parts.pathway.position, size: 3 },
          getColor: { value: parts.pathway.color, size: 3 },
        },
      },
      pointSize: 5.5,
      opacity: 1,
      material: false,
      pickable: true,
      parameters: { depthWriteEnabled: false, depthCompare: 'always' },
      onHover: (info: PickingInfo) => showHover(info, parts.pathway.index),
    }),
  );

  return layers;
}

function showHover(info: PickingInfo, lookup: Uint32Array) {
  const el = document.getElementById('hover')!;
  if (info.index < 0 || !info.picked) {
    el.style.display = 'none';
    return;
  }
  const i = lookup[info.index];
  const type = scene.somata.cellTypes[scene.somata.typeIndex[i]];
  const stage = scene.somata.stages[scene.somata.typeIndex[i]];
  el.innerHTML =
    `<span class="t">${type}</span><br>` +
    `<span class="s">${stage.replace('_', ' ')} &middot; body ${scene.somata.bodyId[i]}</span>`;
  el.style.display = 'block';
  el.style.left = `${info.x + 14}px`;
  el.style.top = `${info.y + 14}px`;
}

function render() {
  deck.setProps({ layers: buildLayers(), viewState });
}

function flyTo(next: Partial<ViewState>) {
  viewState = { ...viewState, ...next };
  render();
}

function presets() {
  const cv = scene.shells.find((s) => s.name === 'CV');
  const neckTarget = cv ? centroid(cv.positions) : ([0, 0, 0] as [number, number, number]);

  window.addEventListener('keydown', (event) => {
    switch (event.key) {
      case '1': // anterior
        flyTo({ target: [0, 0, 0], rotationX: 0, rotationOrbit: 0, zoom: 0.08 });
        break;
      case '2': // lateral
        flyTo({ target: [0, 0, 0], rotationX: 0, rotationOrbit: 90, zoom: 0.08 });
        break;
      case '3': // dorsal
        flyTo({ target: [0, 0, 0], rotationX: 88, rotationOrbit: 0, zoom: 0.08 });
        break;
      case '4': // the neck connective, framed on the giant fiber's descent
        flyTo({ target: neckTarget, rotationX: 6, rotationOrbit: 28, zoom: 1.5 });
        break;
      case 'c':
        showContext = !showContext;
        render();
        break;
      case 's':
        showShells = !showShells;
        render();
        break;
    }
  });
}

function buildLegend() {
  const counts = new Map<string, number>();
  for (let i = 0; i < scene.somata.count; i++) {
    if (!scene.somata.pathway[i]) continue;
    const type = scene.somata.cellTypes[scene.somata.typeIndex[i]];
    counts.set(type, (counts.get(type) ?? 0) + 1);
  }

  const byStage = new Map<string, string[]>();
  for (const type of counts.keys()) {
    const stage = scene.somata.stages[scene.somata.cellTypes.indexOf(type)];
    if (!byStage.has(stage)) byStage.set(stage, []);
    byStage.get(stage)!.push(type);
  }

  const order = ['visual_projection', 'descending', 'motor'];
  const html = order
    .filter((stage) => byStage.has(stage))
    .map((stage) => {
      const rows = byStage
        .get(stage)!
        .sort()
        .map((type) => {
          const color = scene.manifest.palette.cell_type[type] ?? '#64748b';
          return `<div class="row"><span class="sw" style="background:${color}"></span>${type}<span class="n">${counts.get(type)}</span></div>`;
        })
        .join('');
      return `<div class="stage"><div class="stage-name">${stage.replace('_', ' ')}</div>${rows}</div>`;
    })
    .join('');

  document.getElementById('legend-body')!.innerHTML = html;
}

async function main() {
  scene = await loadScene();

  document.getElementById('loading')!.remove();
  for (const id of ['title', 'legend', 'keys', 'stats']) {
    document.getElementById(id)!.hidden = false;
  }
  document.getElementById('subtitle')!.textContent =
    `${scene.manifest.dataset} · ${scene.somata.count.toLocaleString()} somata`;
  document.getElementById('stats')!.innerHTML =
    `${scene.manifest.extent_um.map((v) => Math.round(v)).join(' × ')} µm<br>` +
    `${scene.shells.reduce((a, s) => a + s.n_triangles, 0).toLocaleString()} triangles`;

  buildLegend();
  presets();

  deck = new Deck({
    canvas: 'canvas',
    // Without these the canvas keeps its 300x150 default backing store and deck renders
    // into a postage stamp that the browser then stretches across the window.
    width: '100%',
    height: '100%',
    useDevicePixels: true,
    views: new OrbitView({ orbitAxis: 'Z', fovy: 50 }),
    initialViewState: viewState,
    controller: { inertia: 250 },
    onViewStateChange: ({ viewState: next }: any) => {
      viewState = next;
      render();
    },
    parameters: { clearColor: [0.031, 0.035, 0.043, 1] },
    layers: buildLayers(),
  });
}

main();
