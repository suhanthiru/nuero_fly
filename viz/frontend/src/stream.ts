/**
 * Websocket client for the simulation stream.
 *
 * The server precomputes a trial and plays it back at 20 Hz under an adjustable time
 * dilation, so this is a subscriber rather than a driver: it holds the latest frame and
 * lets the renderer read it whenever it draws. Frames are deliberately not queued - if the
 * renderer falls behind, the right thing is to show the newest state, not to replay a
 * backlog.
 */

export interface History {
  stride: number;
  aggregate: Record<string, number[]>;
  theta_deg?: number[];
  theta_dot?: number[];
}

export interface Hello {
  type: 'hello';
  history: History;
  body_ids: string[];
  cell_types: string[];
  aggregate_types: string[];
  palette: Record<string, string>;
  dt_ms: number;
  duration_ms: number;
  n_steps: number;
  render_hz: number;
  speed: number;
  collision_ms: number | null;
  ratio_ms: number | null;
  events: Record<string, any>;
  meta: Record<string, any>;
}

export interface Frame {
  type: 'frame';
  step: number;
  t_ms: number;
  activity: number[];
  aggregate: Record<string, number>;
  theta_deg?: number;
  theta_dot?: number;
  distance_mm?: number;
}

type Listener = (hello: Hello) => void;

export class Stream {
  hello: Hello | null = null;
  frame: Frame | null = null;
  connected = false;
  busy: string | null = null;

  private socket: WebSocket | null = null;
  private onHello: Listener[] = [];

  constructor(private url: string) {}

  connect(): void {
    this.socket = new WebSocket(this.url);
    this.socket.onopen = () => {
      this.connected = true;
    };
    this.socket.onclose = () => {
      this.connected = false;
      // The server is a local process; a drop usually means it is restarting.
      setTimeout(() => this.connect(), 1500);
    };
    this.socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === 'hello') {
        this.hello = message as Hello;
        this.onHello.forEach((fn) => fn(this.hello!));
      } else if (message.type === 'frame') {
        this.frame = message as Frame;
      } else if (message.type === 'status') {
        this.busy = message.busy ?? null;
      }
    };
  }

  whenReady(fn: Listener): void {
    this.onHello.push(fn);
    if (this.hello) fn(this.hello);
  }

  send(command: Record<string, unknown>): void {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(command));
    }
  }

  play(): void {
    this.send({ cmd: 'play' });
  }
  pause(): void {
    this.send({ cmd: 'pause' });
  }
  seek(step: number): void {
    this.send({ cmd: 'seek', step });
  }
  speed(value: number): void {
    this.send({ cmd: 'speed', value });
  }
  rerun(ratio_ms: number, gain_scale: number, seed: number): void {
    this.send({ cmd: 'rerun', ratio_ms, gain_scale, seed });
  }
}

/**
 * Maps streamed activity onto the neurons the scene already knows about.
 *
 * The stream indexes activity by body id and the scene indexes geometry by its own order,
 * so this builds the crosswalk once rather than searching per frame.
 */
export class ActivityIndex {
  /** activity column for each soma in the scene's pathway subset, or -1 */
  readonly columnForSoma: Int32Array;
  /** activity column for each named cell type's members */
  readonly columnsByType = new Map<string, number[]>();

  constructor(hello: Hello, pathwayBodyIds: BigUint64Array, order: Uint32Array) {
    const column = new Map<string, number>();
    hello.body_ids.forEach((id, i) => column.set(id, i));

    this.columnForSoma = new Int32Array(order.length).fill(-1);
    for (let i = 0; i < order.length; i++) {
      const id = pathwayBodyIds[order[i]].toString();
      const found = column.get(id);
      if (found !== undefined) this.columnForSoma[i] = found;
    }

    hello.cell_types.forEach((type, i) => {
      if (!this.columnsByType.has(type)) this.columnsByType.set(type, []);
      this.columnsByType.get(type)!.push(i);
    });
  }
}
