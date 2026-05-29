import type { MechanicsCategory, ToolSpec } from "./types";

export function toolsForCategory(
  tools: ToolSpec[],
  categoryId: string
): ToolSpec[] {
  return tools
    .filter((tool) => tool.mechanics_category === categoryId)
    .sort(
      (left: ToolSpec, right: ToolSpec) =>
        left.mechanics_stage - right.mechanics_stage ||
        left.label.localeCompare(right.label)
    );
}

export function firstCategoryWithTools(
  categories: MechanicsCategory[],
  tools: ToolSpec[]
): string {
  const category = categories.find(
    (candidate) => toolsForCategory(tools, candidate.id).length > 0
  );
  return category?.id ?? categories[0]?.id ?? "";
}

export function formatJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}
