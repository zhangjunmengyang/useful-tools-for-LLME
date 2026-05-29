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
