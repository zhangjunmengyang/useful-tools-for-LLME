import { useEffect, useRef } from "react";
import type { AdvancedLabProps } from "./types";

export function numberFrom(
  state: Record<string, unknown> | undefined,
  key: string,
  fallback: number,
  min = -Infinity,
  max = Infinity,
) {
  const value = Number(state?.[key]);
  return Number.isFinite(value) ? Math.min(max, Math.max(min, value)) : fallback;
}

export function stringFrom(
  state: Record<string, unknown> | undefined,
  key: string,
  fallback: string,
) {
  return typeof state?.[key] === "string" ? state[key] : fallback;
}

export function booleanFrom(
  state: Record<string, unknown> | undefined,
  key: string,
  fallback: boolean,
) {
  return typeof state?.[key] === "boolean" ? state[key] : fallback;
}

export function pickFrom<T extends string>(
  state: Record<string, unknown> | undefined,
  key: string,
  allowed: readonly T[],
  fallback: T,
): T {
  const value = state?.[key];
  return typeof value === "string" && (allowed as readonly string[]).includes(value)
    ? (value as T)
    : fallback;
}

export function round(value: number, digits = 2) {
  const power = 10 ** digits;
  return Math.round(value * power) / power;
}

export function softmax(values: number[]) {
  const peak = Math.max(...values);
  const exponentials = values.map((item) => Math.exp(item - peak));
  const denominator = exponentials.reduce((sum, item) => sum + item, 0);
  return exponentials.map((item) => item / denominator);
}

export function entropy(probabilities: number[]) {
  return -probabilities.reduce(
    (sum, probability) =>
      probability > 0 ? sum + probability * Math.log(probability) : sum,
    0,
  );
}

export function mean(values: number[]) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export function standardDeviation(values: number[]) {
  const average = mean(values);
  return Math.sqrt(
    mean(values.map((value) => (value - average) * (value - average))),
  );
}

export function sigmoid(value: number) {
  return 1 / (1 + Math.exp(-value));
}

export function cosine(a: readonly number[], b: readonly number[]) {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let index = 0; index < a.length; index += 1) {
    dot += a[index] * b[index];
    normA += a[index] * a[index];
    normB += b[index] * b[index];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

export function polylinePoints(
  values: number[],
  width = 280,
  height = 88,
  yMax?: number,
) {
  const peak = Math.max(1e-6, yMax ?? Math.max(...values, 0));
  if (values.length === 0) return "";
  return values
    .map((value, index) => {
      const x =
        (index / Math.max(1, values.length - 1)) * (width - 10) + 5;
      const y = height - 8 - (Math.max(0, value) / peak) * (height - 16);
      return `${round(x, 1)},${round(y, 1)}`;
    })
    .join(" ");
}

export function useCompletionGate(
  passed: boolean,
  onComplete: AdvancedLabProps["onComplete"],
  payload: Record<string, unknown>,
) {
  const emitted = useRef(false);

  useEffect(() => {
    if (!passed) {
      emitted.current = false;
      return;
    }
    if (!emitted.current) {
      emitted.current = true;
      onComplete?.(payload);
    }
  }, [onComplete, passed, payload]);
}
