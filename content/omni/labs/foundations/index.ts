import type { ComponentType } from "react";
import { Lab01PipelineTracer } from "./Lab01PipelineTracer";
import { Lab02ConnectorWorkbench } from "./Lab02ConnectorWorkbench";
import { Lab03CodecWorkbench } from "./Lab03CodecWorkbench";
import { Lab04MulticodebookScheduler } from "./Lab04MulticodebookScheduler";
import { Lab05StreamingCausality } from "./Lab05StreamingCausality";
import { Lab06TurnPolicyTimeline } from "./Lab06TurnPolicyTimeline";
import { Lab07FullDuplexPolicy } from "./Lab07FullDuplexPolicy";
import { Lab08DynamicTiling } from "./Lab08DynamicTiling";
import { Lab09AVTimeline } from "./Lab09AVTimeline";
import { Lab10TokenPareto } from "./Lab10TokenPareto";
import type { FoundationLabProps } from "./types";

export type { FoundationLabProps } from "./types";

export const foundationLabMap: Record<
  string,
  ComponentType<FoundationLabProps>
> = {
  "01": Lab01PipelineTracer,
  "02": Lab02ConnectorWorkbench,
  "03": Lab03CodecWorkbench,
  "04": Lab04MulticodebookScheduler,
  "05": Lab05StreamingCausality,
  "06": Lab06TurnPolicyTimeline,
  "07": Lab07FullDuplexPolicy,
  "08": Lab08DynamicTiling,
  "09": Lab09AVTimeline,
  "10": Lab10TokenPareto,
};

export type FoundationLabId =
  | "01"
  | "02"
  | "03"
  | "04"
  | "05"
  | "06"
  | "07"
  | "08"
  | "09"
  | "10";
