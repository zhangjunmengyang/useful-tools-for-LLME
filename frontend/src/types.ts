export type LessonMode = "read" | "learn" | "play";

export type TopicSummary = {
  id: string;
  title: string;
  title_en?: string;
  short: string;
  short_en?: string;
  blurb: string;
  blurb_en?: string;
  kind: string;
  ready: boolean;
  source: string;
  note: string;
  modes: LessonMode[];
};

export type OutlineLesson = {
  id: string;
  title: string;
  title_en?: string;
  summary: string;
  summary_en?: string;
  unit_id: string;
  play_tools: string[];
  number?: string;
  slug?: string;
};

export type OutlineUnit = {
  id: string;
  title: string;
  title_en?: string;
  question: string;
  question_en?: string;
  order?: number;
  lessons: OutlineLesson[];
};

export type TopicOutline = {
  id: string;
  title: string;
  title_en?: string;
  blurb: string;
  blurb_en?: string;
  summary: string;
  summary_en?: string;
  ready: boolean;
  source: string;
  original_url?: string | null;
  note?: string;
  units: OutlineUnit[];
  lessons: OutlineLesson[];
  default_lesson_id: string | null;
};

export type LessonDetail = {
  id: string;
  topic_id: string;
  title: string;
  title_en?: string;
  summary: string;
  summary_en?: string;
  unit_id: string;
  format: string;
  read: string;
  learn: string;
  play: string;
  read_en?: string | null;
  learn_en?: string | null;
  play_en?: string | null;
  play_tools: string[];
  checkpoints: string[];
  body_locale?: "zh" | "both";
  original_url?: string | null;
  source_path: string;
};

export type MechanicsCategory = {
  id: string;
  label: string;
  subtitle: string;
  description: string;
  stage: number;
};

export type ToolSpec = {
  id: string;
  label: string;
  description: string;
  lab: string;
  input_schema: Record<string, unknown>;
  output_schema: Record<string, unknown>;
  concepts: string[];
  dependencies: string[];
  requires_model_download: boolean;
  page_id: string | null;
  mechanics_category: string;
  mechanics_stage: number;
  mechanics_category_label: string;
  mechanics_category_subtitle: string;
  sample_input?: Record<string, unknown>;
};

export type ToolsPayload = {
  categories: MechanicsCategory[];
  tools: ToolSpec[];
};

export type ToolRun = {
  tool_id: string;
  status: "success" | "error";
  inputs: Record<string, unknown>;
  result: Record<string, unknown>;
  duration_ms: number;
  error: string | null;
  artifact?: {
    markdown_path: string;
    json_path: string;
  } | null;
  started_at: string;
};

export type LabPage = {
  id: string;
  label: string;
  lab: string;
  lab_label: string;
  group: string;
  group_description: string;
  module: string;
  embed_url: string;
};

export type LabsPayload = {
  mounted: boolean;
  embed_root: string;
  pages: LabPage[];
};
