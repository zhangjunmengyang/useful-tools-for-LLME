import { useEffect, useRef } from "react";

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

export function round(value: number, digits = 2) {
  const power = 10 ** digits;
  return Math.round(value * power) / power;
}

export function softmax(values: number[]) {
  const peak = Math.max(...values);
  const exponentials = values.map((value) => Math.exp(value - peak));
  const denominator = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / denominator);
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

type AdvancedLabProps = import("./types").AdvancedLabProps;
