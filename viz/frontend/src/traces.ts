/**
 * The trace panel: a scrolling window of the stimulus and per-cell-type activity.
 *
 * The 3D view answers *where*. This answers *what*, and it is where the causality is
 * actually legible - LC4 and LPLC2 rise, then the descending neurons, then the motor
 * neuron, on one shared time axis. A 3D brain alone cannot show ordering.
 *
 * The whole trial's series arrives once, up front, and the panel draws a window ending at
 * the playhead. Accumulating samples as frames arrived instead would leave the panel blank
 * for the first several seconds and unable ever to show anything before the moment the
 * client connected.
 *
 * Plain 2D canvas: these are strip charts, and nothing here needs a GPU.
 */

const DIM = '#7b828c';
const INK = '#e6e8ea';
const HAIR = 'rgba(230,232,234,0.10)';

export interface TraceRow {
  key: string;
  colour: string;
  /** value that reaches full row height */
  scale: number;
  /** full-trial series, already decimated by `stride` simulation steps per sample */
  series: number[];
}

export class TracePanel {
  private context: CanvasRenderingContext2D;
  private rows: TraceRow[] = [];
  private stride = 1;
  private windowSamples = 320;
  private markers: { sample: number; label: string }[] = [];

  constructor(private canvas: HTMLCanvasElement) {
    this.context = canvas.getContext('2d')!;
  }

  configure(rows: TraceRow[], stride: number): void {
    this.rows = rows;
    this.stride = Math.max(1, stride);
    this.markers = [];
  }

  /** Mark an instant, given in simulation steps. */
  mark(step: number, label: string): void {
    const sample = Math.round(step / this.stride);
    if (!this.markers.some((m) => m.label === label && m.sample === sample)) {
      this.markers.push({ sample, label });
    }
  }

  clearMarkers(): void {
    this.markers = [];
  }

  /** Draw the window ending at `step` (simulation steps). */
  draw(step: number): void {
    const { canvas, context } = this;
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(1, canvas.clientWidth);
    const height = Math.max(1, canvas.clientHeight);
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    if (!this.rows.length) return;

    const labelWidth = 60;
    const plotLeft = labelWidth;
    const plotWidth = Math.max(10, width - labelWidth - 12);
    const rowHeight = height / this.rows.length;

    const head = Math.round(step / this.stride);
    const start = head - this.windowSamples + 1;
    const xOf = (sample: number) =>
      plotLeft + ((sample - start) / (this.windowSamples - 1)) * plotWidth;

    context.font = '9px ui-monospace, Menlo, Consolas, monospace';
    context.textBaseline = 'middle';

    this.rows.forEach((row, rowIndex) => {
      const top = rowIndex * rowHeight;
      const baseline = top + rowHeight - 3;
      const usable = Math.max(4, rowHeight - 7);

      context.strokeStyle = HAIR;
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(plotLeft, baseline + 1.5);
      context.lineTo(plotLeft + plotWidth, baseline + 1.5);
      context.stroke();

      context.fillStyle = DIM;
      context.fillText(row.key, 4, top + rowHeight / 2);

      // current value, right-aligned against the playhead
      const current = row.series[Math.min(Math.max(head, 0), row.series.length - 1)] ?? 0;
      context.fillStyle = row.colour;
      context.textAlign = 'right';
      context.fillText(
        Math.abs(current) >= 10 ? current.toFixed(0) : current.toFixed(current < 1 ? 3 : 2),
        width - 3,
        top + rowHeight / 2,
      );
      context.textAlign = 'left';

      context.strokeStyle = row.colour;
      context.lineWidth = 1.25;
      context.beginPath();
      let started = false;
      for (let sample = Math.max(0, start); sample <= head; sample++) {
        const value = row.series[sample];
        if (value === undefined) break;
        const x = xOf(sample);
        const y = baseline - Math.max(0, Math.min(1, value / row.scale)) * usable;
        if (!started) {
          context.moveTo(x, y);
          started = true;
        } else {
          context.lineTo(x, y);
        }
      }
      context.stroke();
    });

    // event markers, drawn across every row
    for (const marker of this.markers) {
      if (marker.sample < start || marker.sample > head) continue;
      const x = xOf(marker.sample);
      context.strokeStyle = '#f43f5e';
      context.setLineDash([2, 3]);
      context.beginPath();
      context.moveTo(x, 0);
      context.lineTo(x, height);
      context.stroke();
      context.setLineDash([]);
      context.fillStyle = '#f43f5e';
      context.fillText(marker.label, x + 3, 8);
    }

    // playhead
    context.strokeStyle = INK;
    context.globalAlpha = 0.45;
    context.beginPath();
    context.moveTo(plotLeft + plotWidth, 0);
    context.lineTo(plotLeft + plotWidth, height);
    context.stroke();
    context.globalAlpha = 1;
  }
}
