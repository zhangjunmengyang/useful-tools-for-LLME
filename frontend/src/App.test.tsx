import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";

import App from "./App";
import type { ToolsPayload } from "./types";

const payload: ToolsPayload = {
  categories: [
    {
      id: "input_tokens",
      label: "Input & Tokens",
      subtitle: "Text to model-ready token IDs.",
      description: "Inspect tokenization.",
      stage: 1
    },
    {
      id: "data_context",
      label: "Data & Context",
      subtitle: "Datasets and context before the model.",
      description: "Inspect datasets.",
      stage: 5
    }
  ],
  tools: [
    {
      id: "unicode_analyze",
      label: "Unicode Analysis",
      description: "Inspect Unicode characters.",
      lab: "TokenLab",
      input_schema: { type: "object" },
      output_schema: { type: "object" },
      concepts: ["unicode"],
      dependencies: [],
      requires_model_download: false,
      page_id: "token_playground",
      mechanics_category: "input_tokens",
      mechanics_stage: 2,
      mechanics_category_label: "Input & Tokens",
      mechanics_category_subtitle: "Text to model-ready token IDs."
    },
    {
      id: "dataset_quality_check",
      label: "Dataset Quality Check",
      description: "Check samples for duplicates.",
      lab: "DataLab",
      input_schema: { type: "object" },
      output_schema: { type: "object" },
      concepts: ["data"],
      dependencies: [],
      requires_model_download: false,
      page_id: "data_dataset_viewer",
      mechanics_category: "data_context",
      mechanics_stage: 1,
      mechanics_category_label: "Data & Context",
      mechanics_category_subtitle: "Datasets and context before the model."
    }
  ]
};

describe("App", () => {
  it("renders the Mechanics Explorer shell from tools payload", () => {
    render(<App initialPayload={payload} />);

    expect(
      screen.getByRole("heading", { name: "LLM Mechanics Explorer" })
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Input & Tokens/ })
    ).toBeInTheDocument();
    expect(screen.getByText("Mechanics Canvas")).toBeInTheDocument();
    expect(screen.getByText("Inspector")).toBeInTheDocument();
    expect(
      screen.getByText("POST /api/tools/unicode_analyze/run")
    ).toBeInTheDocument();
    expect(screen.getByText("Response Schema")).toBeInTheDocument();
  });
});
