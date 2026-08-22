/**
 * The arena panel: what the fly sees, plus the geometry producing it.
 *
 * The main view is the fly's eye - a dark disc expanding across the visual field with its
 * subtended angle drawn on it. That is deliberately the *input variable* rather than a
 * third-person view of the chase, because it makes the whole causal chain visible in one
 * glance: the disc grows, theta-dot rises, LC4 and LPLC2 light up, the giant fiber fires.
 * A top-down inset carries the true geometry and closing distance.
 *
 * 2D canvas: the fly's-eye view is a circle whose angular radius is theta/2, so there is no
 * 3D here to speak of.
 */

const DIM = '#7b828c';
const INK = '#e6e8ea';
const PREDATOR = '#1b1e24';
const PREDATOR_EDGE = '#3f4650';

/** Half-width of the rendered visual field. */
const FIELD_HALF_ANGLE_DEG = 90;

export class ArenaPanel {
  private context: CanvasRenderingContext2D;

  constructor(private canvas: HTMLCanvasElement) {
    this.context = canvas.getContext('2d')!;
  }

  draw(state: {
    thetaDeg: number;
    thetaDot: number;
    distanceMm: number;
    startDistanceMm: number;
    tMs: number;
    collisionMs: number | null;
    fired: boolean;
  }): void {
    const { canvas, context } = this;
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (canvas.width !== width * ratio || canvas.height !== height * ratio) {
      canvas.width = width * ratio;
      canvas.height = height * ratio;
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    context.font = '9px ui-monospace, Menlo, Consolas, monospace';
    context.textBaseline = 'middle';

    const insetHeight = 46;
    const eyeHeight = height - insetHeight;

    // ---- fly's eye ---------------------------------------------------------------
    const cx = width / 2;
    const cy = eyeHeight / 2;
    const fieldRadius = Math.min(width, eyeHeight) / 2 - 8;

    // visual field boundary
    context.strokeStyle = 'rgba(230,232,234,0.10)';
    context.lineWidth = 1;
    context.beginPath();
    context.arc(cx, cy, fieldRadius, 0, Math.PI * 2);
    context.stroke();

    // the looming disc, its angular radius mapped linearly onto the field
    const halfAngle = Math.min(state.thetaDeg / 2, FIELD_HALF_ANGLE_DEG);
    const discRadius = (halfAngle / FIELD_HALF_ANGLE_DEG) * fieldRadius;
    if (discRadius > 0.4) {
      context.fillStyle = PREDATOR;
      context.strokeStyle = state.fired ? '#f43f5e' : PREDATOR_EDGE;
      context.lineWidth = state.fired ? 1.6 : 1;
      context.beginPath();
      context.arc(cx, cy, discRadius, 0, Math.PI * 2);
      context.fill();
      context.stroke();

      // subtended-angle annotation
      context.strokeStyle = DIM;
      context.globalAlpha = 0.7;
      context.beginPath();
      context.moveTo(cx - discRadius, cy);
      context.lineTo(cx + discRadius, cy);
      context.stroke();
      context.globalAlpha = 1;
      context.fillStyle = DIM;
      context.textAlign = 'center';
      context.fillText(`${state.thetaDeg.toFixed(0)}°`, cx, cy - discRadius - 8);
      context.textAlign = 'left';
    }

    context.fillStyle = DIM;
    context.fillText("fly's eye", 4, 8);

    // ---- top-down inset ----------------------------------------------------------
    const top = eyeHeight;
    context.strokeStyle = 'rgba(230,232,234,0.10)';
    context.beginPath();
    context.moveTo(0, top + 0.5);
    context.lineTo(width, top + 0.5);
    context.stroke();

    const flyY = top + insetHeight - 12;
    const flyX = width / 2;

    // the fly
    context.fillStyle = INK;
    context.beginPath();
    context.arc(flyX, flyY, 2.5, 0, Math.PI * 2);
    context.fill();

    // approach track and current position
    const trackTop = top + 12;
    const fraction = Math.max(
      0,
      Math.min(1, 1 - state.distanceMm / Math.max(state.startDistanceMm, 1e-6)),
    );
    context.strokeStyle = 'rgba(230,232,234,0.12)';
    context.beginPath();
    context.moveTo(flyX, trackTop);
    context.lineTo(flyX, flyY);
    context.stroke();

    const predatorY = trackTop + fraction * (flyY - trackTop);
    context.fillStyle = PREDATOR_EDGE;
    context.beginPath();
    context.arc(flyX, predatorY, 4, 0, Math.PI * 2);
    context.fill();

    context.fillStyle = DIM;
    context.fillText(`${state.distanceMm.toFixed(0)} mm`, 4, top + 12);
    const remaining =
      state.collisionMs === null ? null : state.collisionMs - state.tMs;
    if (remaining !== null) {
      context.textAlign = 'right';
      context.fillText(
        remaining >= 0 ? `contact in ${remaining.toFixed(0)} ms` : 'after contact',
        width - 4,
        top + 12,
      );
      context.textAlign = 'left';
    }
  }
}
