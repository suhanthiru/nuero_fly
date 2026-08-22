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

export interface Scene {
  manifest: Manifest;
  somata: Somata;
  shells: Shell[];
}

export async function loadScene(base = '/scene'): Promise<Scene> {
  const manifest: Manifest = await (await fetch(`${base}/manifest.json`)).json();

  const [somataBuf, shellBuf] = await Promise.all([
    fetch(`${base}/${manifest.somata.file}`).then((r) => r.arrayBuffer()),
    fetch(`${base}/${manifest.compartments.file}`).then((r) => r.arrayBuffer()),
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

  return { manifest, somata, shells };
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
