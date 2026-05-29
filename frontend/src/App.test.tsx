import "@testing-library/jest-dom/vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

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
      input_schema: { type: "object", properties: { text: { type: "string" } } },
      output_schema: { type: "object", properties: { chars: { type: "array" } } },
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
      id: "token_count",
      label: "Token Count",
      description: "Count tokenizer output IDs.",
      lab: "TokenLab",
      input_schema: {
        type: "object",
        properties: { model_name: { type: "string" } }
      },
      output_schema: {
        type: "object",
        properties: { token_count: { type: "number" } }
      },
      concepts: ["tokens"],
      dependencies: [],
      requires_model_download: false,
      page_id: "token_playground",
      mechanics_category: "input_tokens",
      mechanics_stage: 3,
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
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("skips fetching tools when initialPayload is provided", () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    render(<App initialPayload={payload} />);

    expect(
      screen.getByRole("heading", { name: "LLM Mechanics Explorer" })
    ).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("fetches tools and renders fetched category and tool data without initialPayload", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(payload)
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    expect(screen.getByText("Loading tools from /api/tools")).toBeInTheDocument();
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/tools"));
    expect(
      screen.getByRole("button", { name: /Input & Tokens/ })
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Unicode Analysis/ })).toBeInTheDocument();
  });

  it("updates canvas and API drawer when a category is selected", () => {
    render(<App initialPayload={payload} />);

    fireEvent.click(screen.getByRole("button", { name: /Data & Context/ }));

    expect(
      screen.getByRole("heading", { name: "Data & Context" })
    ).toBeInTheDocument();
    expect(screen.getByText("Inspect datasets.")).toBeInTheDocument();
    expect(
      screen.getByText("POST /api/tools/dataset_quality_check/run")
    ).toBeInTheDocument();
  });

  it("updates inspector endpoint and schema when a tool is selected", () => {
    render(<App initialPayload={payload} />);

    fireEvent.click(screen.getByRole("button", { name: /Token Count/ }));

    expect(
      screen.getByText("POST /api/tools/token_count/run")
    ).toBeInTheDocument();
    expect(screen.getByText(/model_name/)).toBeInTheDocument();
    expect(screen.getAllByText(/token_count/)).toHaveLength(2);
  });

  it("syncs selected category and tool when initialPayload changes", () => {
    const nextPayload: ToolsPayload = {
      categories: [payload.categories[1]],
      tools: [payload.tools[2]]
    };

    const { rerender } = render(<App initialPayload={payload} />);

    rerender(<App initialPayload={nextPayload} />);

    expect(
      screen.getByRole("heading", { name: "Data & Context" })
    ).toBeInTheDocument();
    expect(screen.getByText("Mechanics Canvas")).toBeInTheDocument();
    expect(screen.getByText("Inspector")).toBeInTheDocument();
    expect(
      screen.getByText("POST /api/tools/dataset_quality_check/run")
    ).toBeInTheDocument();
    expect(screen.getByText("Response Schema")).toBeInTheDocument();
  });
});
