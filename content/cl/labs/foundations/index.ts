import type { ComponentType } from "react";
import { Lab01ForgettingSlider } from "./Lab01ForgettingSlider";
import { Lab02StabilityPlane } from "./Lab02StabilityPlane";
import { Lab03MetricLiar } from "./Lab03MetricLiar";
import { Lab04CallXiaowang } from "./Lab04CallXiaowang";
import { Lab05FisherPins } from "./Lab05FisherPins";
import { Lab06ReplayBackpack } from "./Lab06ReplayBackpack";
import { Lab07PacknetWall } from "./Lab07PacknetWall";
import { Lab08GradientProjection } from "./Lab08GradientProjection";
import { Lab09DataMix } from "./Lab09DataMix";
import { Lab10TaskHeatmap } from "./Lab10TaskHeatmap";
import { Lab11LoraOrthogonal } from "./Lab11LoraOrthogonal";
import { Lab12TaskVectorAdd } from "./Lab12TaskVectorAdd";
import type { FoundationLabProps } from "./types";

export type { FoundationLabProps } from "./types";

export const foundationLabMap: Record<
  string,
  ComponentType<FoundationLabProps>
> = {
  "01": Lab01ForgettingSlider,
  "02": Lab02StabilityPlane,
  "03": Lab03MetricLiar,
  "04": Lab04CallXiaowang,
  "05": Lab05FisherPins,
  "06": Lab06ReplayBackpack,
  "07": Lab07PacknetWall,
  "08": Lab08GradientProjection,
  "09": Lab09DataMix,
  "10": Lab10TaskHeatmap,
  "11": Lab11LoraOrthogonal,
  "12": Lab12TaskVectorAdd,
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
  | "10"
  | "11"
  | "12";
