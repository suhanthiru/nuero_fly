/**
 * Loads the scene assets written by scripts/build_scene.py.
 *
 * The binary is a flat concatenation of typed arrays described by a layout in the
 * manifest. Deliberately not glTF: the frontend wants arrays it can hand straight to the
 * GPU, and a hand-rolled layout is far less code than a glTF writer plus a parser.
 */

export interface ArraySpec {
  offset: number;
  length: number;
  dtype: string;
  shape: number[];
}

export interface ShellSpec {
  name: string;
  color: [number, number, number];
  opacity: number;
  n_vertices: number;
  n_triangles: number;
}

export interface MorphologyGroup {
  name: string;
  kind: 'skeleton' | 'mesh';
  color: [number, number, number];
  n_neurons?: number;
  n_requested?: number;
  n_segments?: number;
  bodies?: { body_id: number; n_vertices: number; n_triangles: number }[];
}

export interface Manifest {
  dataset: string;
  citation: string;
  units: string;
  orientation: string;
  extent_um: [number, number, number];
  background: string;
  somata: {
    count: number;
    file: string;
    layout: Record<string, ArraySpec>;
    cell_types: string[];
    stages: string[];
    pathway_count: number;
  };
  compartments: {
    file: string;
    layout: Record<string, ArraySpec>;
    shells: ShellSpec[];
  };
  morphology: {
    file: string;
    layout: Record<string, ArraySpec>;
    groups: MorphologyGroup[];
  };
  palette: {
    stage: Record<string, string>;
    cell_type: Record<string, string>;
  };
}

const CTORS: Record<string, any> = {
  float32: Float32Array,
  uint32: Uint32Array,
  uint16: Uint16Array,
  uint8: Uint8Array,
  uint64: BigUint64Array,
};

function view(buffer: ArrayBuffer, spec: ArraySpec) {
  const Ctor = CTORS[spec.dtype];
  if (!Ctor) throw new Error(`unsupported dtype ${spec.dtype}`);
  return new Ctor(buffer, spec.offset, spec.length / Ctor.BYTES_PER_ELEMENT);
}

export interface Somata {
  count: number;
  position: Float32Array;
  color: Uint8Array;
  pathway: Uint8Array;
  typeIndex: Uint16Array;
  bodyId: BigUint64Array;
  cellTypes: string[];
  stages: string[];
}

export interface Shell extends ShellSpec {
  positions: Float32Array;
  indices: Uint32Array;
}

export interface SkeletonBundle {
  name: string;
  color: [number, number, number];
  count: number;
  source: Float32Array;
  target: Float32Array;
  owner: Uint16Array;
  neurons: number;
}

export interface NeuronMesh {
  name: string;
  bodyId: number;
  color: [number, number, number];
  positions: Float32Array;
  indices: Uint32Array;
}

export interface Scene {
  manifest: Manifest;
  somata: Somata;
  shells: Shell[];
  bundles: SkeletonBundle[];
  neuronMeshes: NeuronMesh[];
}

export async function loadScene(base = '/scene'): Promise<Scene> {
  const manifest: Manifest = await (await fetch(`${base}/manifest.json`)).json();

  const [somataBuf, shellBuf, morphBuf] = await Promise.all([
    fetch(`${base}/${manifest.somata.file}`).then((r) => r.arrayBuffer()),
    fetch(`${base}/${manifest.compartments.file}`).then((r) => r.arrayBuffer()),
    fetch(`${base}/${manifest.morphology.file}`).then((r) => r.arrayBuffer()),
  ]);

  const sl = manifest.somata.layout;
  const somata: Somata = {
    count: manifest.somata.count,
    position: view(somataBuf, sl.position),
    color: view(somataBuf, sl.color),
    pathway: view(somataBuf, sl.pathway),
    typeIndex: view(somataBuf, sl.type_index),
    bodyId: view(somataBuf, sl.body_id),
    cellTypes: manifest.somata.cell_types,
    stages: manifest.somata.stages,
  };

  const cl = manifest.compartments.layout;
  const shells: Shell[] = manifest.compartments.shells.map((spec) => ({
    ...spec,
    positions: view(shellBuf, cl[`${spec.name}/position`]),
    indices: view(shellBuf, cl[`${spec.name}/index`]),
  }));

  const ml = manifest.morphology.layout;
  const bundles: SkeletonBundle[] = [];
  const neuronMeshes: NeuronMesh[] = [];

  for (const group of manifest.morphology.groups) {
    if (group.kind === 'skeleton') {
      const owner = view(morphBuf, ml[`${group.name}/owner`]);
      bundles.push({
        name: group.name,
        color: group.color,
        count: owner.length,
        source: view(morphBuf, ml[`${group.name}/source`]),
        target: view(morphBuf, ml[`${group.name}/target`]),
        owner,
        neurons: group.n_neurons ?? 0,
      });
    } else {
      for (const body of group.bodies ?? []) {
        neuronMeshes.push({
          name: group.name,
          bodyId: body.body_id,
          color: group.color,
          positions: view(morphBuf, ml[`${group.name}/${body.body_id}/position`]),
          indices: view(morphBuf, ml[`${group.name}/${body.body_id}/index`]),
        });
      }
    }
  }

  return { manifest, somata, shells, bundles, neuronMeshes };
}

/**
 * Split the point cloud into context and pathway subsets.
 *
 * They are drawn as separate layers because deck.gl's PointCloudLayer takes one size for
 * all its points, and the whole design rests on pathway neurons being larger and brighter
 * than the anatomical context they sit in.
 */
export function partition(somata: Somata) {
  const n = somata.count;
  let pathwayCount = 0;
  for (let i = 0; i < n; i++) if (somata.pathway[i]) pathwayCount++;
  const contextCount = n - pathwayCount;

  const out = {
    context: {
      count: contextCount,
      position: new Float32Array(contextCount * 3),
      color: new Uint8Array(contextCount * 3),
      index: new Uint32Array(contextCount),
    },
    pathway: {
      count: pathwayCount,
      position: new Float32Array(pathwayCount * 3),
      color: new Uint8Array(pathwayCount * 3),
      index: new Uint32Array(pathwayCount),
    },
  };

  let c = 0;
  let p = 0;
  for (let i = 0; i < n; i++) {
    const target = somata.pathway[i] ? out.pathway : out.context;
    const j = somata.pathway[i] ? p++ : c++;
    target.position[j * 3] = somata.position[i * 3];
    target.position[j * 3 + 1] = somata.position[i * 3 + 1];
    target.position[j * 3 + 2] = somata.position[i * 3 + 2];
    target.color[j * 3] = somata.color[i * 3];
    target.color[j * 3 + 1] = somata.color[i * 3 + 1];
    target.color[j * 3 + 2] = somata.color[i * 3 + 2];
    target.index[j] = i;
  }
  return out;
}
