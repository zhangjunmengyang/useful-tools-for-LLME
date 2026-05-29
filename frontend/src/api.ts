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
