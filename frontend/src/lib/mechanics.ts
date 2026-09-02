import type { MechanicsCategory, ToolSpec } from "../types";

export function toolsForCategory(tools: ToolSpec[], categoryId: string): ToolSpec[] {
  return tools
    .filter((tool) => tool.mechanics_category === categoryId)
    .sort(
      (left: ToolSpec, right: ToolSpec) =>
        left.mechanics_stage - right.mechanics_stage || left.label.localeCompare(right.label),
    );
}

export function firstCategoryWithTools(categories: MechanicsCategory[], tools: ToolSpec[]): string {
  const category = categories.find((candidate) => toolsForCategory(tools, candidate.id).length > 0);
  return category?.id ?? categories[0]?.id ?? "";
}

export function formatJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

export function buildExampleInput(tool?: ToolSpec): string {
  if (!tool) return "{}";
  const sampleInput = asRecord(tool.sample_input);
  if (Object.keys(sampleInput).length > 0) {
    return formatJson(sampleInput);
  }
  return formatJson(buildSchemaExample(tool));
}

export function buildCurlCommand(toolId: string, jsonInput: string): string {
  const endpoint = new URL(`/api/tools/${toolId}/run`, window.location.origin).toString();
  return ["curl -X POST", shellQuote(endpoint), "-H 'Content-Type: application/json'", `-d ${shellQuote(jsonInput)}`].join(
    " ",
  );
}

export async function copyCurlCommand(toolId: string, jsonInput: string): Promise<void> {
  await navigator.clipboard?.writeText(buildCurlCommand(toolId, jsonInput));
}

function shellQuote(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`;
}

function buildSchemaExample(tool: ToolSpec): Record<string, unknown> {
  const schema = tool.input_schema;
  const properties = asRecord(schema.properties);
  const required = Array.isArray(schema.required)
    ? schema.required.filter((key): key is string => typeof key === "string")
    : Object.keys(properties).slice(0, 4);
  const example: Record<string, unknown> = {};
  for (const key of required) {
    example[key] = exampleValueForProperty(key, asRecord(properties[key]));
  }
  return example;
}

function exampleValueForProperty(key: string, property: Record<string, unknown>): unknown {
  const enumValues = Array.isArray(property.enum) ? property.enum : [];
  if (enumValues.length > 0) return enumValues[0];
  switch (property.type) {
    case "array":
      return [exampleValueForProperty(key, asRecord(property.items))];
    case "boolean":
      return false;
    case "integer":
    case "number":
      return 1;
    case "object":
      return {};
    case "string":
    default:
      if (key.includes("model")) return "gpt2";
      if (key.includes("text")) return "Ａ café";
      return "example";
  }
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}
