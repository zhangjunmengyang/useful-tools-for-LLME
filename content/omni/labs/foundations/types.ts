export type FoundationLabProps = {
  onComplete?: (state?: Record<string, unknown>) => void;
  initialState?: Record<string, unknown>;
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
