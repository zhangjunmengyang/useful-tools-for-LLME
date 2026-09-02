export type DiagramNodeKind =
  | "input"
  | "transform"
  | "state"
  | "decision"
  | "output";

export interface DiagramPoint {
  x: number;
  y: number;
}

export interface DiagramNode {
  id: string;
  label: readonly string[];
  meta?: string;
  kind: DiagramNodeKind;
  x: number;
  y: number;
  width?: number;
  height?: number;
}

export interface DiagramEdge {
  id: string;
  from: string;
  to: string;
  label?: string;
  via?: readonly DiagramPoint[];
  labelAt?: DiagramPoint;
}

export interface DiagramStep {
  title: string;
  description: string;
  focus: readonly string[];
}

export interface LessonDiagram {
  lessonId: string;
  title: string;
  summary: string;
  viewBox?: string;
  nodes: readonly DiagramNode[];
  edges: readonly DiagramEdge[];
  steps: readonly DiagramStep[];
  facts: readonly string[];
}
