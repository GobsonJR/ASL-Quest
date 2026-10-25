// Pure helpers for the landmark debug view (no DOM, no MediaPipe) so they can be unit-tested.

/** Rolling window of recent samples with median / p95, used for latency readouts. */
export class RollingStat {
  private values: number[] = [];
  constructor(private readonly size = 120) {}

  add(value: number): void {
    this.values.push(value);
    if (this.values.length > this.size) this.values.shift();
  }

  get count(): number {
    return this.values.length;
  }

  quantile(q: number): number | null {
    if (!this.values.length) return null;
    const sorted = [...this.values].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
  }

  get median(): number | null {
    return this.quantile(0.5);
  }

  get p95(): number | null {
    return this.quantile(0.95);
  }

  reset(): void {
    this.values = [];
  }
}

/** Frames per second over the last `windowMs` of frame timestamps (ms). */
export class FpsMeter {
  private stamps: number[] = [];
  constructor(private readonly windowMs = 1000) {}

  tick(nowMs: number): number {
    this.stamps.push(nowMs);
    while (this.stamps.length > 1 && nowMs - this.stamps[0] > this.windowMs) this.stamps.shift();
    if (this.stamps.length < 2) return 0;
    return ((this.stamps.length - 1) * 1000) / (nowMs - this.stamps[0]);
  }

  reset(): void {
    this.stamps = [];
  }
}

export type Point3 = { x: number; y: number; z: number };

/** Mean per-point displacement (in normalised image units) between two consecutive detections: a stability measure.
 * A still hand should give a value near 0; jitter shows up as a persistent non-zero floor. */
export function meanJitter(prev: Point3[], next: Point3[]): number | null {
  if (prev.length !== next.length || !prev.length) return null;
  let sum = 0;
  for (let i = 0; i < prev.length; i += 1) sum += Math.hypot(next[i].x - prev[i].x, next[i].y - prev[i].y);
  return sum / prev.length;
}

/** The body of a /predict_landmarks request for a detection made on an UNMIRRORED video frame. */
export function landmarkPayload(points: Point3[], handedness: string | null, imageMirrored = false) {
  return {
    landmarks: points.map((p) => [p.x, p.y, p.z]),
    handedness: (handedness === "Left" || handedness === "Right" ? handedness : null) as "Left" | "Right" | null,
    image_mirrored: imageMirrored,
  };
}
