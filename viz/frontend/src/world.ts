/**
 * The 3D world: a fly, a floor, and a predator closing on it.
 *
 * Same renderer and same clock as the brain view, so the two are literally two cameras on
 * one simulation - step N here is step N of the spike train. Everything is millimetres,
 * which is the scale the physics runs at; the brain view uses micrometres and is a separate
 * coordinate space entirely.
 *
 * The fly is drawn at true scale, using the flybody geometry (Vaxenburg et al., Nature
 * 2025). That is worth stating because it looks wrong at first: a Drosophila is ~2.5 mm long
 * against a 20 mm stimulus, so it really is a speck. The camera frames a few centimetres
 * around it rather than the whole 160 mm approach.
 *
 * The body is a SHELL. Its legs and wings are frozen in the model's resting pose and are
 * never actuated - the escape here is a rigid-body impulse, as it has been since Phase 3.
 * A detailed fly whose joints never move risks implying leg mechanics we do not simulate,
 * which is why the HUD labels it.
 *
 * The predator is a flat disc rather than a solid, because that is the stimulus the
 * behavioural literature presents and the shape the encoder's angular-size geometry assumes.
 */

import { LineLayer, PathLayer, PointCloudLayer, ScatterplotLayer } from '@deck.gl/layers';
import { SimpleMeshLayer } from '@deck.gl/mesh-layers';

const FLOOR = '#171b21';
const GRID = [42, 48, 58] as const;
const PREDATOR = [190, 64, 76] as const;
const FLY_RESTING = [230, 190, 90] as const;
const FLY_LAUNCHED = [120, 220, 255] as const;
const TRAIL = [120, 140, 170] as const;

export interface World {
  stride: number;
  fly: number[][];
  predator: number[][];
  predator_radius_mm: number;
  fly_size_mm: [number, number, number];
  takeoff_step: number | null;
  escaped: boolean;
  closest_approach_mm: number;
  azimuth_deg: number;
  heading_deg: number;
}

/** A UV sphere as flat arrays, since deck.gl wants raw attributes. */
function sphere(radius: number, segments = 24): {
  positions: Float32Array;
  indices: Uint32Array;
} {
  const positions: number[] = [];
  const indices: number[] = [];
  for (let ring = 0; ring <= segments; ring++) {
    const phi = (ring / segments) * Math.PI;
    for (let seg = 0; seg <= segments * 2; seg++) {
      const theta = (seg / (segments * 2)) * Math.PI * 2;
      positions.push(
        radius * Math.sin(phi) * Math.cos(theta),
        radius * Math.sin(phi) * Math.sin(theta),
        radius * Math.cos(phi),
      );
    }
  }
  const perRing = segments * 2 + 1;
  for (let ring = 0; ring < segments; ring++) {
    for (let seg = 0; seg < segments * 2; seg++) {
      const a = ring * perRing + seg;
      const b = a + perRing;
      indices.push(a, b, a + 1, a + 1, b, b + 1);
    }
  }
  return { positions: new Float32Array(positions), indices: new Uint32Array(indices) };
}

/** An ellipsoid, the fallback body if the baked fly mesh is unavailable. */
function ellipsoid(rx: number, ry: number, rz: number, segments = 14) {
  const unit = sphere(1, segments);
  const positions = new Float32Array(unit.positions.length);
  for (let i = 0; i < unit.positions.length; i += 3) {
    positions[i] = unit.positions[i] * rx;
    positions[i + 1] = unit.positions[i + 1] * ry;
    positions[i + 2] = unit.positions[i + 2] * rz;
  }
  return { positions, indices: unit.indices };
}

/**
 * A flat disc standing perpendicular to the approach bearing.
 *
 * This is the stimulus the looming literature actually presents - a dark disc expanding on
 * a screen, not a solid object - and it is the shape the encoder's angular-size geometry
 * assumes, since theta = 2 arctan(r/d) is the disc form. The rotation is baked into the
 * vertices rather than passed as an orientation, because the bearing is fixed for a trial
 * and baking it avoids depending on deck.gl's Euler convention.
 */
function disc(radius: number, azimuthDeg: number, segments = 64) {
  const a = (azimuthDeg * Math.PI) / 180;
  // Normal along the bearing; the disc spans the two perpendicular directions.
  const ux = -Math.sin(a);
  const uy = Math.cos(a);
  const positions: number[] = [0, 0, 0];
  const indices: number[] = [];
  for (let i = 0; i <= segments; i++) {
    const t = (i / segments) * Math.PI * 2;
    positions.push(
      radius * Math.cos(t) * ux,
      radius * Math.cos(t) * uy,
      radius * Math.sin(t),
    );
  }
  for (let i = 1; i <= segments; i++) indices.push(0, i, i + 1);
  return { positions: new Float32Array(positions), indices: new Uint32Array(indices) };
}

function floorGrid(extent: number, spacing: number) {
  const source: number[][] = [];
  const target: number[][] = [];
  for (let v = -extent; v <= extent; v += spacing) {
    source.push([-extent, v, 0], [v, -extent, 0]);
    target.push([extent, v, 0], [v, extent, 0]);
  }
  return source.map((s, i) => ({ source: s, target: target[i] }));
}

export interface BakedMesh {
  positions: Float32Array;
  indices: Uint32Array;
}

export class WorldScene {
  private predatorMesh: { positions: Float32Array; indices: Uint32Array };
  private flyMesh: { positions: Float32Array; indices: Uint32Array };
  private grid: { source: number[]; target: number[] }[];
  /** True when the body is the real flybody geometry rather than the fallback blob. */
  readonly hasBody: boolean;

  constructor(private world: World, flyMesh?: BakedMesh | null) {
    this.predatorMesh = disc(world.predator_radius_mm, world.azimuth_deg);
    const [rx, ry, rz] = world.fly_size_mm;
    this.flyMesh = flyMesh ?? ellipsoid(rx, ry, rz, 14);
    this.hasBody = Boolean(flyMesh);
    this.grid = floorGrid(120, 10);
  }

  /** Convert a simulation step to an index into the decimated paths. */
  private sample(step: number): number {
    const index = Math.round(step / this.world.stride);
    return Math.max(0, Math.min(index, this.world.fly.length - 1));
  }

  flyPosition(step: number): number[] {
    return this.world.fly[this.sample(step)];
  }

  hasLaunched(step: number): boolean {
    return this.world.takeoff_step !== null && step >= this.world.takeoff_step;
  }

  layers(step: number): any[] {
    const index = this.sample(step);
    const fly = this.world.fly[index];
    const predator = this.world.predator[index];
    const launched = this.hasLaunched(step);

    // Trails: where each body has been up to now. The fly's is the visible record of the
    // escape, so it only exists once it has left the ground.
    const flyTrail = this.world.fly.slice(0, index + 1);
    const predatorTrail = this.world.predator.slice(0, index + 1);

    const layers: any[] = [
      new LineLayer({
        id: 'world-grid',
        data: this.grid,
        getSourcePosition: (d: any) => d.source,
        getTargetPosition: (d: any) => d.target,
        getColor: [...GRID, 90] as any,
        getWidth: 1,
        widthMinPixels: 1,
        parameters: { depthWriteEnabled: false },
      }),
      // Ground shadows, so height is readable. Without them the fly's arc is ambiguous.
      new ScatterplotLayer({
        id: 'world-shadows',
        data: [
          { position: [predator[0], predator[1], 0.05], radius: this.world.predator_radius_mm },
          { position: [fly[0], fly[1], 0.05], radius: 2.4 },
        ],
        getPosition: (d: any) => d.position,
        getRadius: (d: any) => d.radius,
        radiusUnits: 'common',
        getFillColor: [0, 0, 0, 130] as any,
        parameters: { depthWriteEnabled: false },
      }),
      new PathLayer({
        id: 'world-predator-trail',
        data: [{ path: predatorTrail }],
        getPath: (d: any) => d.path,
        getColor: [...PREDATOR, 90] as any,
        getWidth: 0.7,
        widthUnits: 'common',
        widthMinPixels: 1,
        parameters: { depthWriteEnabled: false },
      }),
      new SimpleMeshLayer({
        id: 'world-predator',
        data: [{ position: predator }],
        mesh: {
          attributes: { POSITION: { value: this.predatorMesh.positions, size: 3 } },
          indices: { value: this.predatorMesh.indices, size: 1 },
        },
        getPosition: (d: any) => d.position,
        getColor: [...PREDATOR, 225] as any,
        material: false,
        // A disc has no back, so it must not be culled.
        parameters: { depthWriteEnabled: true, cullMode: 'none' },
      }),
    ];

    if (launched && flyTrail.length > 1) {
      layers.push(
        new PathLayer({
          id: 'world-fly-trail',
          data: [{ path: flyTrail }],
          getPath: (d: any) => d.path,
          getColor: [...FLY_LAUNCHED, 170] as any,
          getWidth: 0.35,
          widthUnits: 'common',
          widthMinPixels: 1.5,
          parameters: { depthWriteEnabled: false },
        }),
      );
    }

    layers.push(
      new SimpleMeshLayer({
        id: 'world-fly',
        data: [{ position: fly }],
        mesh: {
          attributes: { POSITION: { value: this.flyMesh.positions, size: 3 } },
          indices: { value: this.flyMesh.indices, size: 1 },
        },
        getPosition: (d: any) => d.position,
        getColor: [...(launched ? FLY_LAUNCHED : FLY_RESTING), 255] as any,
        material: false,
        updateTriggers: { getColor: launched },
        parameters: { depthWriteEnabled: true, cullMode: 'back' },
      }),
      // A marker at the fly, so it stays findable when the camera is pulled back - at true
      // scale a 2.5 mm body is a couple of pixels across from any useful distance.
      new PointCloudLayer({
        id: 'world-fly-marker',
        data: [{ position: fly }],
        getPosition: (d: any) => d.position,
        getColor: [...(launched ? FLY_LAUNCHED : FLY_RESTING), 190] as any,
        pointSize: 7,
        material: false,
        parameters: { depthWriteEnabled: false, depthCompare: 'always' },
      }),
    );

    return layers;
  }
}
