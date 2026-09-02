export type FoundationLabProps = {
  onComplete?: (state?: Record<string, unknown>) => void;
  initialState?: Record<string, unknown>;
};

export type Point2 = {
  x: number;
  y: number;
  cls: number;
};

export function initialNumber(
  state: Record<string, unknown> | undefined,
  key: string,
  fallback: number,
) {
  const value = state?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

export function initialString<T extends string>(
  state: Record<string, unknown> | undefined,
  key: string,
  allowed: readonly T[],
  fallback: T,
) {
  const value = state?.[key];
  return typeof value === "string" && allowed.includes(value as T)
    ? (value as T)
    : fallback;
}

export function initialBoolean(
  state: Record<string, unknown> | undefined,
  key: string,
  fallback: boolean,
) {
  const value = state?.[key];
  return typeof value === "boolean" ? value : fallback;
}

/** Mulberry32. Keep this exact; labs 01/02 lock seed 11, lab 06 locks seed 7. */
export function createRng(seed: number) {
  let a = seed | 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(rng: () => number) {
  const u = Math.max(rng(), 1e-9);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function makeBlobs(
  rng: () => number,
  count: number,
  mean0: readonly [number, number],
  mean1: readonly [number, number],
  std: number,
  class0 = 0,
  class1 = 1,
): Point2[] {
  const points: Point2[] = [];
  const half = Math.floor(count / 2);
  for (let i = 0; i < count; i += 1) {
    const isOne = i >= half;
    const mean = isOne ? mean1 : mean0;
    points.push({
      x: mean[0] + std * gaussian(rng),
      y: mean[1] + std * gaussian(rng),
      cls: isOne ? class1 : class0,
    });
  }
  return points;
}

export function sigmoid(z: number) {
  const clipped = Math.max(-25, Math.min(25, z));
  return 1 / (1 + Math.exp(-clipped));
}

export function softmax(logits: number[]) {
  const max = Math.max(...logits);
  const exps = logits.map((z) => Math.exp(z - max));
  const sum = exps.reduce((acc, value) => acc + value, 0);
  return exps.map((value) => value / sum);
}

export function logisticAcc(
  weights: readonly [number, number],
  bias: number,
  data: readonly Point2[],
) {
  if (data.length === 0) return 0;
  let correct = 0;
  for (const point of data) {
    const score = weights[0] * point.x + weights[1] * point.y + bias;
    const predicted = score >= 0 ? 1 : 0;
    if (predicted === point.cls) correct += 1;
  }
  return correct / data.length;
}

export function logisticStep(
  weights: readonly [number, number],
  bias: number,
  batch: readonly Point2[],
  lr: number,
  weightDecay: number,
): { weights: [number, number]; bias: number } {
  let g0 = 0;
  let g1 = 0;
  let gb = 0;
  for (const point of batch) {
    const score = weights[0] * point.x + weights[1] * point.y + bias;
    const err = sigmoid(score) - point.cls;
    g0 += err * point.x;
    g1 += err * point.y;
    gb += err;
  }
  const n = batch.length || 1;
  return {
    weights: [
      weights[0] - lr * (g0 / n + weightDecay * weights[0]),
      weights[1] - lr * (g1 / n + weightDecay * weights[1]),
    ],
    bias: bias - lr * (gb / n + weightDecay * bias),
  };
}

export function formatPct(value: number) {
  return `${Math.round(value * 100)}%`;
}

export function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}
