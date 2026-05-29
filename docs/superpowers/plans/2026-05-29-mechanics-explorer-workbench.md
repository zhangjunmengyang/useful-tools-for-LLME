# Mechanics Explorer Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first stateless React + FastAPI Mechanics Explorer workbench around the approved Pipeline Rail taxonomy.

**Architecture:** Keep Python research logic in `workbench_tools` as the source of truth. Add a FastAPI service that exposes tool metadata, stateless run endpoints, and explicit artifact export endpoints. Add a Vite React frontend that consumes those endpoints and renders the three-column Pipeline Rail, Canvas, and Inspector experience without depending on Gradio.

**Tech Stack:** Python, FastAPI, unittest, Vite, React, TypeScript, Vitest, CSS.

---

## File Structure

- Create `workbench_tools/mechanics.py`: owns the seven Pipeline Rail categories and enriches existing `ToolSpec` objects with workbench metadata.
- Modify `workbench_tools/schemas.py`: add optional `mechanics_category` and `mechanics_stage` fields to `ToolSpec`.
- Modify `workbench_tools/builtin_tools.py`: assign each existing tool to one of the approved categories.
- Create `workbench_api/__init__.py`: package marker for the HTTP API.
- Create `workbench_api/app.py`: FastAPI app and stateless endpoints.
- Create `tests/test_mechanics_taxonomy.py`: verifies all categories and all registered tools have valid mechanics metadata.
- Create `tests/test_workbench_api.py`: verifies stateless API behavior through FastAPI `TestClient`.
- Modify `requirements.txt`: add FastAPI runtime and test dependencies.
- Create `frontend/package.json`: Vite/React scripts and dependencies.
- Create `frontend/index.html`: React mount point.
- Create `frontend/src/main.tsx`: app entrypoint.
- Create `frontend/src/App.tsx`: shell composition and client-side state.
- Create `frontend/src/api.ts`: typed API client.
- Create `frontend/src/types.ts`: shared frontend types matching API payloads.
- Create `frontend/src/mechanics.ts`: frontend category labels and display helpers derived from API data.
- Create `frontend/src/App.css`: product layout and responsive CSS.
- Create `frontend/src/App.test.tsx`: render-level tests for rail, canvas, inspector, and API drawer.
- Create `frontend/vite.config.ts`: Vite + Vitest config.
- Create `frontend/tsconfig.json`: TypeScript config.
- Create `frontend/tsconfig.node.json`: Node-side TypeScript config for Vite.
- Modify `README.md`: add commands for the new API and frontend while leaving the Gradio command documented as legacy.

## Task 1: Mechanics Taxonomy Metadata

**Files:**
- Modify: `workbench_tools/schemas.py`
- Modify: `workbench_tools/builtin_tools.py`
- Create: `workbench_tools/mechanics.py`
- Create: `tests/test_mechanics_taxonomy.py`

- [ ] **Step 1: Write failing taxonomy tests**

Add `tests/test_mechanics_taxonomy.py`:

```python
"""Mechanics Explorer taxonomy tests."""

from __future__ import annotations

import unittest

from workbench_tools.mechanics import (
    MECHANICS_CATEGORIES,
    VALID_MECHANICS_CATEGORY_IDS,
    enrich_tool_spec,
)
from workbench_tools.registry import get_registry


class MechanicsTaxonomyTest(unittest.TestCase):
    """确保工具注册表能映射到 Mechanics Explorer 的一级导航。"""

    def test_pipeline_rail_has_approved_categories(self):
        category_ids = [category["id"] for category in MECHANICS_CATEGORIES]

        self.assertEqual(
            category_ids,
            [
                "input_tokens",
                "representation_space",
                "probability_decoding",
                "transformer_anatomy",
                "data_context",
                "adaptation_cost",
                "evaluation_traces",
            ],
        )

    def test_every_registered_tool_has_valid_mechanics_metadata(self):
        for spec in get_registry().list_specs():
            with self.subTest(tool_id=spec.id):
                self.assertIsNotNone(spec.mechanics_category)
                self.assertIn(spec.mechanics_category, VALID_MECHANICS_CATEGORY_IDS)
                self.assertIsInstance(spec.mechanics_stage, int)
                self.assertGreaterEqual(spec.mechanics_stage, 1)

    def test_enriched_tool_spec_includes_category_details(self):
        spec = get_registry().get_spec("dataset_quality_check")
        payload = enrich_tool_spec(spec)

        self.assertEqual(payload["mechanics_category"], "data_context")
        self.assertEqual(payload["mechanics_category_label"], "Data & Context")
        self.assertIn("input_schema", payload)
        self.assertIn("output_schema", payload)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
.venv/bin/python -m unittest tests.test_mechanics_taxonomy -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'workbench_tools.mechanics'` or `AttributeError` for missing `mechanics_category`.

- [ ] **Step 3: Add optional mechanics fields to ToolSpec**

In `workbench_tools/schemas.py`, extend `ToolSpec`:

```python
@dataclass(slots=True)
class ToolSpec:
    """描述一个可由 UI、CLI 或外部 Agent 调用的研究工具。"""

    id: str
    label: str
    description: str
    lab: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    concepts: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    requires_model_download: bool = False
    page_id: str | None = None
    mechanics_category: str | None = None
    mechanics_stage: int = 999

    def to_dict(self) -> dict[str, Any]:
        """返回 JSON 兼容字典。"""
        return make_json_safe(self)
```

- [ ] **Step 4: Create taxonomy helpers**

Create `workbench_tools/mechanics.py`:

```python
"""Mechanics Explorer taxonomy helpers."""

from __future__ import annotations

from typing import Any

from .schemas import ToolSpec


MECHANICS_CATEGORIES: list[dict[str, Any]] = [
    {
        "id": "input_tokens",
        "label": "Input & Tokens",
        "subtitle": "Text to model-ready token IDs.",
        "description": "Inspect tokenization, Unicode normalization, compression, and chat template rendering.",
        "stage": 1,
    },
    {
        "id": "representation_space",
        "label": "Representation Space",
        "subtitle": "Vectors, similarity, and latent geometry.",
        "description": "Explore embedding spaces, vector arithmetic, semantic similarity, and sparse-versus-dense behavior.",
        "stage": 2,
    },
    {
        "id": "probability_decoding",
        "label": "Probability & Decoding",
        "subtitle": "Logits to next-token decisions.",
        "description": "Inspect logits, sampling controls, top-k, top-p, temperature, and beam search behavior.",
        "stage": 3,
    },
    {
        "id": "transformer_anatomy",
        "label": "Transformer Anatomy",
        "subtitle": "Attention, RoPE, FFN, and KV cache.",
        "description": "Visualize transformer internals and inference-time memory mechanics.",
        "stage": 4,
    },
    {
        "id": "data_context",
        "label": "Data & Context",
        "subtitle": "Datasets and context before the model.",
        "description": "Inspect datasets, cleaning, formatting, chunking, and retrieval diagnostics.",
        "stage": 5,
    },
    {
        "id": "adaptation_cost",
        "label": "Adaptation & Cost",
        "subtitle": "Fine-tuning, memory, and budget.",
        "description": "Estimate LoRA parameters, training cost, model memory, and configuration differences.",
        "stage": 6,
    },
    {
        "id": "evaluation_traces",
        "label": "Evaluation & Traces",
        "subtitle": "Metrics, judges, and run behavior.",
        "description": "Evaluate predictions and inspect model or agent traces.",
        "stage": 7,
    },
]

CATEGORY_BY_ID = {category["id"]: category for category in MECHANICS_CATEGORIES}
VALID_MECHANICS_CATEGORY_IDS = set(CATEGORY_BY_ID)


def enrich_tool_spec(spec: ToolSpec) -> dict[str, Any]:
    """返回带 Mechanics Explorer 分类信息的工具定义。"""
    payload = spec.to_dict()
    category = CATEGORY_BY_ID.get(spec.mechanics_category or "")
    if category:
        payload["mechanics_category_label"] = category["label"]
        payload["mechanics_category_subtitle"] = category["subtitle"]
    else:
        payload["mechanics_category_label"] = "Uncategorized"
        payload["mechanics_category_subtitle"] = "No mechanics category assigned."
    return payload
```

- [ ] **Step 5: Assign categories in built-in tools**

In `workbench_tools/builtin_tools.py`, add these fields to the matching `ToolSpec(...)` calls:

```python
# eval_metrics
mechanics_category="evaluation_traces",
mechanics_stage=1,

# tokenizer_encode
mechanics_category="input_tokens",
mechanics_stage=1,

# unicode_analyze
mechanics_category="input_tokens",
mechanics_stage=2,

# sampling_distribution
mechanics_category="probability_decoding",
mechanics_stage=1,

# kv_cache_growth
mechanics_category="transformer_anatomy",
mechanics_stage=4,

# rope_frequencies
mechanics_category="transformer_anatomy",
mechanics_stage=2,

# ffn_activation_compare
mechanics_category="transformer_anatomy",
mechanics_stage=3,

# data_clean
mechanics_category="data_context",
mechanics_stage=3,

# dataset_quality_check
mechanics_category="data_context",
mechanics_stage=1,

# instruct_format
mechanics_category="data_context",
mechanics_stage=4,

# kv_cache_estimate
mechanics_category="transformer_anatomy",
mechanics_stage=5,

# lora_params_estimate
mechanics_category="adaptation_cost",
mechanics_stage=1,

# training_cost_estimate
mechanics_category="adaptation_cost",
mechanics_stage=2,

# rag_chunk
mechanics_category="data_context",
mechanics_stage=5,

# rag_lexical_retrieval
mechanics_category="data_context",
mechanics_stage=6,

# trace_analyze
mechanics_category="evaluation_traces",
mechanics_stage=2,
```

- [ ] **Step 6: Run taxonomy and existing toolbox tests**

Run:

```bash
.venv/bin/python -m unittest tests.test_mechanics_taxonomy tests.test_research_toolbox -v
```

Expected: PASS.

- [ ] **Step 7: Commit taxonomy changes**

Run:

```bash
git add workbench_tools/schemas.py workbench_tools/builtin_tools.py workbench_tools/mechanics.py tests/test_mechanics_taxonomy.py
git commit -m "feat: add mechanics taxonomy metadata"
```

## Task 2: Stateless FastAPI Service

**Files:**
- Modify: `requirements.txt`
- Create: `workbench_api/__init__.py`
- Create: `workbench_api/app.py`
- Create: `tests/test_workbench_api.py`

- [ ] **Step 1: Write failing API tests**

Create `tests/test_workbench_api.py`:

```python
"""Stateless HTTP API tests for Mechanics Explorer."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from workbench_api.app import app


class WorkbenchApiTest(unittest.TestCase):
    """验证 FastAPI 服务暴露普通 JSON 工具接口。"""

    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_tools_endpoint_returns_categories_and_tools(self):
        response = self.client.get("/api/tools")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("categories", payload)
        self.assertIn("tools", payload)
        self.assertIn("data_context", {category["id"] for category in payload["categories"]})
        dataset_tool = next(tool for tool in payload["tools"] if tool["id"] == "dataset_quality_check")
        self.assertEqual(dataset_tool["mechanics_category"], "data_context")
        self.assertEqual(dataset_tool["mechanics_category_label"], "Data & Context")

    def test_run_endpoint_is_stateless_and_returns_result_immediately(self):
        response = self.client.post(
            "/api/tools/unicode_analyze/run",
            json={"text": "Ａ café"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["tool_id"], "unicode_analyze")
        self.assertEqual(payload["status"], "success")
        self.assertIn("result", payload)
        self.assertNotIn("run_id", payload)
        self.assertFalse(payload["result"]["normalization"]["nfkc_equal"])

    def test_run_endpoint_returns_validation_error_without_exception_page(self):
        response = self.client.post(
            "/api/tools/eval_metrics/run",
            json={"predictions": ["Paris"]},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "error")
        self.assertIn("references", payload["error"])

    def test_export_endpoint_writes_artifact_only_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            response = self.client.post(
                "/api/tools/eval_metrics/export",
                json={
                    "inputs": {"predictions": ["Paris"], "references": ["Paris"]},
                    "output_dir": tmp,
                },
            )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["status"], "success")
            self.assertIn("artifact", payload)
            self.assertTrue(Path(payload["artifact"]["json_path"]).is_file())
            self.assertTrue(Path(payload["artifact"]["markdown_path"]).is_file())

    def test_unknown_tool_returns_404(self):
        response = self.client.post("/api/tools/not_real/run", json={})

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Unknown tool: not_real")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
.venv/bin/python -m unittest tests.test_workbench_api -v
```

Expected: FAIL because `fastapi` or `workbench_api.app` is not available.

- [ ] **Step 3: Add API dependencies**

Add this section to `requirements.txt` under Web Framework:

```text
fastapi>=0.111.0
uvicorn[standard]>=0.30.0
httpx>=0.27.0
```

- [ ] **Step 4: Create API package marker**

Create `workbench_api/__init__.py`:

```python
"""HTTP API for the Mechanics Explorer workbench."""
```

- [ ] **Step 5: Implement FastAPI app**

Create `workbench_api/app.py`:

```python
"""Stateless HTTP API for Mechanics Explorer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from workbench_tools.mechanics import MECHANICS_CATEGORIES, enrich_tool_spec
from workbench_tools.registry import get_registry


class ExportRequest(BaseModel):
    """显式 artifact 导出请求。"""

    inputs: dict[str, Any] = Field(default_factory=dict)
    output_dir: str = "research"


app = FastAPI(
    title="LLM Tools Workbench API",
    version="0.1.0",
)


def _registry():
    """返回工具注册表。"""
    return get_registry()


@app.get("/api/health")
def health() -> dict[str, str]:
    """健康检查。"""
    return {"status": "ok"}


@app.get("/api/tools")
def list_tools() -> dict[str, Any]:
    """返回 Mechanics Explorer 分类和工具元数据。"""
    specs = sorted(
        _registry().list_specs(),
        key=lambda spec: (spec.mechanics_stage, spec.label.lower()),
    )
    return {
        "categories": MECHANICS_CATEGORIES,
        "tools": [enrich_tool_spec(spec) for spec in specs],
    }


@app.get("/api/tools/{tool_id}")
def inspect_tool(tool_id: str) -> dict[str, Any]:
    """返回单个工具元数据。"""
    try:
        spec = _registry().get_spec(tool_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_id}") from exc
    return enrich_tool_spec(spec)


@app.post("/api/tools/{tool_id}/run")
def run_tool(tool_id: str, inputs: dict[str, Any]) -> dict[str, Any]:
    """运行一个 stateless 工具。"""
    try:
        _registry().get_spec(tool_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_id}") from exc
    run = _registry().run(tool_id, inputs, export=False)
    return run.to_dict()


@app.post("/api/tools/{tool_id}/export")
def export_tool(tool_id: str, request: ExportRequest) -> dict[str, Any]:
    """运行工具并显式导出 artifact。"""
    try:
        _registry().get_spec(tool_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_id}") from exc
    output_dir = Path(request.output_dir)
    run = _registry().run(tool_id, request.inputs, export=True, output_dir=output_dir)
    return run.to_dict()
```

- [ ] **Step 6: Run API tests**

Run:

```bash
.venv/bin/python -m unittest tests.test_workbench_api -v
```

Expected: PASS.

- [ ] **Step 7: Run API smoke command**

Run:

```bash
.venv/bin/python -m uvicorn workbench_api.app:app --host 127.0.0.1 --port 8001
```

In another shell:

```bash
curl -sS http://127.0.0.1:8001/api/tools | python -m json.tool | sed -n '1,40p'
```

Expected: JSON starts with `categories` and `tools`. Stop uvicorn with `Ctrl-C`.

- [ ] **Step 8: Commit API changes**

Run:

```bash
git add requirements.txt workbench_api tests/test_workbench_api.py
git commit -m "feat: expose stateless workbench api"
```

## Task 3: React Workbench Skeleton

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/api.ts`
- Create: `frontend/src/types.ts`
- Create: `frontend/src/mechanics.ts`
- Create: `frontend/src/App.css`
- Create: `frontend/src/App.test.tsx`
- Create: `frontend/vite.config.ts`
- Create: `frontend/tsconfig.json`
- Create: `frontend/tsconfig.node.json`

- [ ] **Step 1: Create package manifest**

Create `frontend/package.json`:

```json
{
  "name": "llm-tools-workbench-frontend",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite --host 127.0.0.1 --port 5173",
    "build": "tsc && vite build",
    "test": "vitest run",
    "preview": "vite preview --host 127.0.0.1 --port 4173"
  },
  "dependencies": {
    "@vitejs/plugin-react": "^4.3.0",
    "vite": "^5.4.0",
    "typescript": "^5.5.0",
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@testing-library/jest-dom": "^6.4.0",
    "@testing-library/react": "^15.0.0",
    "@types/react": "^18.3.0",
    "@types/react-dom": "^18.3.0",
    "jsdom": "^24.1.0",
    "vitest": "^2.0.0"
  }
}
```

- [ ] **Step 2: Create TypeScript configs**

Create `frontend/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["DOM", "DOM.Iterable", "ES2020"],
    "allowJs": false,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "module": "ESNext",
    "moduleResolution": "Node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx"
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

Create `frontend/tsconfig.node.json`:

```json
{
  "compilerOptions": {
    "composite": true,
    "module": "ESNext",
    "moduleResolution": "Node",
    "allowSyntheticDefaultImports": true
  },
  "include": ["vite.config.ts"]
}
```

- [ ] **Step 3: Create Vite config**

Create `frontend/vite.config.ts`:

```typescript
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8001"
    }
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: []
  }
});
```

- [ ] **Step 4: Create frontend tests**

Create `frontend/src/App.test.tsx`:

```typescript
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
  it("renders Pipeline Rail, Canvas, and Inspector", () => {
    render(<App initialPayload={payload} />);

    expect(screen.getByText("LLM Mechanics Explorer")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Input & Tokens/i })).toBeInTheDocument();
    expect(screen.getByText("Mechanics Canvas")).toBeInTheDocument();
    expect(screen.getByText("Inspector")).toBeInTheDocument();
  });

  it("shows the API drawer for the selected tool", () => {
    render(<App initialPayload={payload} />);

    expect(screen.getByText("POST /api/tools/unicode_analyze/run")).toBeInTheDocument();
    expect(screen.getByText("Response Schema")).toBeInTheDocument();
  });
});
```

Expected initially: FAIL because `App`, `types`, and supporting files do not exist.

- [ ] **Step 5: Create index and entrypoint**

Create `frontend/index.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>LLM Mechanics Explorer</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

Create `frontend/src/main.tsx`:

```typescript
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./App.css";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

- [ ] **Step 6: Create frontend types**

Create `frontend/src/types.ts`:

```typescript
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
```

- [ ] **Step 7: Create API client**

Create `frontend/src/api.ts`:

```typescript
import type { ToolRun, ToolsPayload } from "./types";

export async function fetchTools(): Promise<ToolsPayload> {
  const response = await fetch("/api/tools");
  if (!response.ok) {
    throw new Error(`Failed to load tools: ${response.status}`);
  }
  return response.json();
}

export async function runTool(
  toolId: string,
  inputs: Record<string, unknown>
): Promise<ToolRun> {
  const response = await fetch(`/api/tools/${toolId}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(inputs)
  });
  if (!response.ok) {
    throw new Error(`Failed to run tool: ${response.status}`);
  }
  return response.json();
}
```

- [ ] **Step 8: Create frontend helpers**

Create `frontend/src/mechanics.ts`:

```typescript
import type { MechanicsCategory, ToolSpec } from "./types";

export function toolsForCategory(
  tools: ToolSpec[],
  categoryId: string
): ToolSpec[] {
  return tools
    .filter((tool) => tool.mechanics_category === categoryId)
    .sort((a, b) => a.mechanics_stage - b.mechanics_stage || a.label.localeCompare(b.label));
}

export function firstCategoryWithTools(
  categories: MechanicsCategory[],
  tools: ToolSpec[]
): string {
  const match = categories.find((category) => toolsForCategory(tools, category.id).length > 0);
  return match?.id ?? categories[0]?.id ?? "";
}

export function formatJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}
```

- [ ] **Step 9: Create React app shell**

Create `frontend/src/App.tsx`:

```typescript
import { useEffect, useMemo, useState } from "react";
import { fetchTools } from "./api";
import { firstCategoryWithTools, formatJson, toolsForCategory } from "./mechanics";
import type { ToolSpec, ToolsPayload } from "./types";

type AppProps = {
  initialPayload?: ToolsPayload;
};

const fallbackPayload: ToolsPayload = {
  categories: [],
  tools: []
};

export default function App({ initialPayload }: AppProps) {
  const [payload, setPayload] = useState<ToolsPayload>(initialPayload ?? fallbackPayload);
  const [loading, setLoading] = useState(!initialPayload);
  const [error, setError] = useState<string | null>(null);
  const [selectedCategoryId, setSelectedCategoryId] = useState("");
  const [selectedToolId, setSelectedToolId] = useState("");

  useEffect(() => {
    if (initialPayload) {
      return;
    }
    fetchTools()
      .then((data) => {
        setPayload(data);
        setSelectedCategoryId(firstCategoryWithTools(data.categories, data.tools));
      })
      .catch((caught: Error) => setError(caught.message))
      .finally(() => setLoading(false));
  }, [initialPayload]);

  useEffect(() => {
    if (!initialPayload) {
      return;
    }
    setSelectedCategoryId(firstCategoryWithTools(initialPayload.categories, initialPayload.tools));
  }, [initialPayload]);

  const selectedCategory = payload.categories.find((category) => category.id === selectedCategoryId);
  const categoryTools = useMemo(
    () => toolsForCategory(payload.tools, selectedCategoryId),
    [payload.tools, selectedCategoryId]
  );
  const selectedTool: ToolSpec | undefined =
    categoryTools.find((tool) => tool.id === selectedToolId) ?? categoryTools[0];

  useEffect(() => {
    if (categoryTools[0] && selectedToolId !== categoryTools[0].id) {
      setSelectedToolId(categoryTools[0].id);
    }
  }, [categoryTools, selectedToolId]);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Stateless API Workbench</p>
          <h1>LLM Mechanics Explorer</h1>
        </div>
        <span className="status-pill">React + FastAPI</span>
      </header>

      {loading && <div className="notice">Loading tools from /api/tools</div>}
      {error && <div className="notice error">API error: {error}</div>}

      <section className="workspace-grid">
        <nav className="pipeline-rail" aria-label="Pipeline Rail">
          {payload.categories.map((category) => (
            <button
              className={category.id === selectedCategoryId ? "rail-item active" : "rail-item"}
              key={category.id}
              onClick={() => setSelectedCategoryId(category.id)}
              type="button"
            >
              <span>{String(category.stage).padStart(2, "0")}</span>
              <strong>{category.label}</strong>
              <small>{category.subtitle}</small>
            </button>
          ))}
        </nav>

        <section className="canvas-panel">
          <p className="eyebrow">Mechanics Canvas</p>
          <h2>{selectedCategory?.label ?? "No category selected"}</h2>
          <p className="lede">{selectedCategory?.description ?? "Start the API to load tools."}</p>
          <div className="tool-grid">
            {categoryTools.map((tool) => (
              <button
                className={tool.id === selectedTool?.id ? "tool-card active" : "tool-card"}
                key={tool.id}
                onClick={() => setSelectedToolId(tool.id)}
                type="button"
              >
                <strong>{tool.label}</strong>
                <span>{tool.description}</span>
              </button>
            ))}
          </div>
          <div className="canvas-empty">
            <strong>{selectedTool?.label ?? "Select a tool"}</strong>
            <span>
              This panel will render the selected mechanism visualization after the run controls are connected.
            </span>
          </div>
        </section>

        <aside className="inspector-panel">
          <p className="eyebrow">Inspector</p>
          <h2>{selectedTool?.label ?? "Tool Inspector"}</h2>
          <p>{selectedTool?.description ?? "No tool selected."}</p>
          {selectedTool && (
            <div className="api-drawer">
              <h3>API</h3>
              <code>POST /api/tools/{selectedTool.id}/run</code>
              <h3>Request Schema</h3>
              <pre>{formatJson(selectedTool.input_schema)}</pre>
              <h3>Response Schema</h3>
              <pre>{formatJson(selectedTool.output_schema)}</pre>
            </div>
          )}
        </aside>
      </section>
    </main>
  );
}
```

- [ ] **Step 10: Create CSS**

Create `frontend/src/App.css`:

```css
:root {
  --bg: #ffffff;
  --surface: #fafafa;
  --surface-strong: #f5f5f5;
  --ink: #171717;
  --muted: #525252;
  --meta: #8a8a8a;
  --border: #e9e9e9;
  --blue: #0070f3;
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--bg);
}

* {
  box-sizing: border-box;
  letter-spacing: 0;
}

body {
  margin: 0;
  background: var(--bg);
}

button,
input,
textarea {
  font: inherit;
}

.app-shell {
  min-height: 100vh;
  padding: 18px 22px 28px;
}

.topbar {
  align-items: center;
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  gap: 16px;
  padding-bottom: 14px;
}

.topbar h1,
.canvas-panel h2,
.inspector-panel h2 {
  font-size: 20px;
  line-height: 1.2;
  margin: 0;
}

.eyebrow {
  color: var(--meta);
  font-size: 11px;
  font-weight: 700;
  margin: 0 0 6px;
  text-transform: uppercase;
}

.status-pill {
  border: 1px solid var(--border);
  border-radius: 999px;
  color: var(--muted);
  font-size: 12px;
  padding: 6px 10px;
}

.notice {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  margin: 14px 0 0;
  padding: 10px 12px;
}

.notice.error {
  border-color: #ee0000;
  color: #a40000;
}

.workspace-grid {
  display: grid;
  gap: 14px;
  grid-template-columns: 260px minmax(0, 1fr) 340px;
  margin-top: 16px;
}

.pipeline-rail,
.canvas-panel,
.inspector-panel {
  min-width: 0;
}

.pipeline-rail {
  border-right: 1px solid var(--border);
  display: grid;
  gap: 8px;
  padding-right: 12px;
}

.rail-item,
.tool-card {
  background: var(--surface);
  border: 1px solid transparent;
  border-radius: 8px;
  color: var(--ink);
  cursor: pointer;
  display: grid;
  gap: 4px;
  padding: 10px;
  text-align: left;
}

.rail-item span,
.rail-item small,
.tool-card span {
  color: var(--muted);
  font-size: 12px;
  line-height: 1.4;
}

.rail-item.active,
.tool-card.active {
  background: var(--ink);
  color: #ffffff;
}

.rail-item.active span,
.rail-item.active small,
.tool-card.active span {
  color: #d4d4d4;
}

.canvas-panel,
.inspector-panel {
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px;
}

.lede {
  color: var(--muted);
  line-height: 1.5;
  margin: 8px 0 16px;
}

.tool-grid {
  display: grid;
  gap: 10px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.canvas-empty {
  align-items: center;
  background: var(--surface-strong);
  border: 1px dashed var(--border);
  border-radius: 10px;
  color: var(--muted);
  display: grid;
  gap: 8px;
  justify-items: center;
  margin-top: 16px;
  min-height: 220px;
  padding: 24px;
  text-align: center;
}

.canvas-empty strong {
  color: var(--ink);
}

.inspector-panel p {
  color: var(--muted);
  line-height: 1.5;
}

.api-drawer {
  border-top: 1px solid var(--border);
  margin-top: 14px;
  padding-top: 14px;
}

.api-drawer h3 {
  font-size: 13px;
  margin: 14px 0 8px;
}

.api-drawer code,
.api-drawer pre {
  background: var(--surface-strong);
  border: 1px solid var(--border);
  border-radius: 8px;
  display: block;
  font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  overflow-x: auto;
  padding: 10px;
}

@media (max-width: 960px) {
  .workspace-grid {
    grid-template-columns: 1fr;
  }

  .pipeline-rail {
    border-right: 0;
    border-bottom: 1px solid var(--border);
    grid-template-columns: repeat(2, minmax(0, 1fr));
    padding: 0 0 12px;
  }

  .tool-grid {
    grid-template-columns: 1fr;
  }
}
```

- [ ] **Step 11: Install and run frontend tests**

Run:

```bash
cd frontend
npm install
npm test
```

Expected: PASS.

- [ ] **Step 12: Build frontend**

Run:

```bash
cd frontend
npm run build
```

Expected: PASS and creates `frontend/dist`.

- [ ] **Step 13: Commit frontend skeleton**

Run:

```bash
git add frontend
git commit -m "feat: add react mechanics explorer shell"
```

## Task 4: Connect Real Tool Runs in the Frontend

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.test.tsx`

- [ ] **Step 1: Add failing run interaction test**

Append this test to `frontend/src/App.test.tsx`:

```typescript
import { fireEvent, waitFor } from "@testing-library/react";

it("runs the selected stateless tool and shows result JSON", async () => {
  const fetchMock = vi.spyOn(global, "fetch").mockResolvedValueOnce(
    new Response(
      JSON.stringify({
        tool_id: "unicode_analyze",
        status: "success",
        inputs: { text: "Ａ café" },
        result: { char_count: 6 },
        duration_ms: 1,
        error: null,
        started_at: "2026-05-29T00:00:00+00:00"
      }),
      { status: 200, headers: { "Content-Type": "application/json" } }
    )
  );

  render(<App initialPayload={payload} />);
  fireEvent.change(screen.getByLabelText("JSON Input"), {
    target: { value: "{\"text\":\"Ａ café\"}" }
  });
  fireEvent.click(screen.getByRole("button", { name: "Run Tool" }));

  await waitFor(() => expect(screen.getByText("\"char_count\": 6")).toBeInTheDocument());
  fetchMock.mockRestore();
});
```

Expected initially: FAIL because there is no JSON input, run button, or result rendering.

- [ ] **Step 2: Update API client if needed**

Confirm `frontend/src/api.ts` still contains:

```typescript
export async function runTool(
  toolId: string,
  inputs: Record<string, unknown>
): Promise<ToolRun> {
  const response = await fetch(`/api/tools/${toolId}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(inputs)
  });
  if (!response.ok) {
    throw new Error(`Failed to run tool: ${response.status}`);
  }
  return response.json();
}
```

- [ ] **Step 3: Wire run controls in App**

In `frontend/src/App.tsx`, import `runTool` and add state near the existing state declarations:

```typescript
import { fetchTools, runTool } from "./api";
import type { ToolRun, ToolSpec, ToolsPayload } from "./types";
```

```typescript
const [jsonInput, setJsonInput] = useState("{\n  \"text\": \"Ａ café\"\n}");
const [runResult, setRunResult] = useState<ToolRun | null>(null);
const [runError, setRunError] = useState<string | null>(null);
const [running, setRunning] = useState(false);
```

Add this handler inside `App`:

```typescript
async function handleRunTool() {
  if (!selectedTool) {
    return;
  }
  setRunning(true);
  setRunError(null);
  try {
    const inputs = JSON.parse(jsonInput) as Record<string, unknown>;
    const result = await runTool(selectedTool.id, inputs);
    setRunResult(result);
  } catch (caught) {
    const message = caught instanceof Error ? caught.message : "Unknown run error";
    setRunError(message);
  } finally {
    setRunning(false);
  }
}
```

Add this controls block to `.inspector-panel` before `.api-drawer`:

```tsx
<label className="input-label" htmlFor="json-input">
  JSON Input
</label>
<textarea
  id="json-input"
  aria-label="JSON Input"
  value={jsonInput}
  onChange={(event) => setJsonInput(event.target.value)}
  rows={8}
/>
<button className="run-button" disabled={!selectedTool || running} onClick={handleRunTool} type="button">
  {running ? "Running" : "Run Tool"}
</button>
{runError && <div className="notice error">{runError}</div>}
```

Replace the `.canvas-empty` body with result-aware rendering:

```tsx
<div className="canvas-empty">
  {runResult ? (
    <pre>{formatJson(runResult.result)}</pre>
  ) : (
    <>
      <strong>{selectedTool?.label ?? "Select a tool"}</strong>
      <span>
        Run the selected tool to render its mechanism output. Results are kept in client state only.
      </span>
    </>
  )}
</div>
```

- [ ] **Step 4: Add CSS for controls and result JSON**

Append to `frontend/src/App.css`:

```css
.input-label {
  color: var(--muted);
  display: block;
  font-size: 12px;
  font-weight: 600;
  margin: 14px 0 6px;
}

textarea {
  background: var(--surface-strong);
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--ink);
  font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  min-height: 150px;
  padding: 10px;
  resize: vertical;
  width: 100%;
}

.run-button {
  background: var(--ink);
  border: 1px solid var(--ink);
  border-radius: 8px;
  color: #ffffff;
  cursor: pointer;
  margin-top: 10px;
  min-height: 40px;
  width: 100%;
}

.run-button:disabled {
  cursor: not-allowed;
  opacity: 0.6;
}

.canvas-empty pre {
  color: var(--ink);
  font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  margin: 0;
  max-width: 100%;
  overflow-x: auto;
  text-align: left;
  width: 100%;
}
```

- [ ] **Step 5: Run frontend tests and build**

Run:

```bash
cd frontend
npm test
npm run build
```

Expected: PASS.

- [ ] **Step 6: Commit run wiring**

Run:

```bash
git add frontend/src/App.tsx frontend/src/App.css frontend/src/api.ts frontend/src/App.test.tsx
git commit -m "feat: run stateless tools from react workbench"
```

## Task 5: Documentation and End-to-End Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README commands**

Add this section after Quick Start in `README.md`:

```markdown
## Mechanics Explorer Preview

The new Mechanics Explorer is a React + FastAPI workbench. It is API-first and stateless: tool runs return immediately, and the server does not keep run history.

Start the API:

```bash
uvicorn workbench_api.app:app --host 127.0.0.1 --port 8001
```

Start the React workbench:

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

Useful API calls:

```bash
curl -sS http://127.0.0.1:8001/api/tools | python -m json.tool
curl -sS -X POST http://127.0.0.1:8001/api/tools/unicode_analyze/run \
  -H 'Content-Type: application/json' \
  -d '{"text":"Ａ café"}' | python -m json.tool
```

The Gradio app remains available through `python app_gradio.py` while the React workbench is being built out.
```

- [ ] **Step 2: Run backend test suite**

Run:

```bash
.venv/bin/python -m unittest tests.test_mechanics_taxonomy tests.test_workbench_api tests.test_research_toolbox -v
```

Expected: PASS.

- [ ] **Step 3: Run Python syntax check**

Run:

```bash
.venv/bin/python -m compileall workbench_tools workbench_api tests
```

Expected: PASS.

- [ ] **Step 4: Run frontend verification**

Run:

```bash
cd frontend
npm test
npm run build
```

Expected: PASS.

- [ ] **Step 5: Run local API smoke test**

Run API server:

```bash
.venv/bin/python -m uvicorn workbench_api.app:app --host 127.0.0.1 --port 8001
```

Run smoke call:

```bash
curl -sS -X POST http://127.0.0.1:8001/api/tools/unicode_analyze/run \
  -H 'Content-Type: application/json' \
  -d '{"text":"Ａ café"}' | python -m json.tool
```

Expected output includes:

```json
{
  "tool_id": "unicode_analyze",
  "status": "success"
}
```

- [ ] **Step 6: Run local frontend smoke test**

Run frontend:

```bash
cd frontend
npm run dev
```

Open `http://127.0.0.1:5173`, select `Input & Tokens`, run `Unicode Analysis` with:

```json
{
  "text": "Ａ café"
}
```

Expected: result JSON appears in the Mechanics Canvas and the Inspector shows `POST /api/tools/unicode_analyze/run`.

- [ ] **Step 7: Commit docs and verification updates**

Run:

```bash
git add README.md
git commit -m "docs: document mechanics explorer preview"
```

## Self-Review Notes

- Spec coverage: taxonomy, React shell, stateless FastAPI, no MCP, no server-side run history, API drawer, and Data & Context dataset boundary are all covered.
- Scope control: first version connects real tool metadata and stateless runs, but does not attempt Gradio parity or every custom visualization.
- Type consistency: backend uses existing `ToolSpec` and `ToolRun`; frontend mirrors those payloads in `types.ts`.
- Verification: backend unit tests, Python compile check, frontend tests, frontend build, API smoke, and browser smoke are included.
