import type { ComponentType } from "react";

export type AdvancedLabProps = {
  onComplete?: (state?: Record<string, unknown>) => void;
  initialState?: Record<string, unknown>;
};

export type AdvancedLabMap = Record<string, ComponentType<AdvancedLabProps>>;
