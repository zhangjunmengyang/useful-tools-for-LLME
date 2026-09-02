import { lessonDiagramCatalog } from "@/lib/lesson-diagrams/catalog";
import { nodeSize } from "@/lib/lesson-diagrams/layout";
import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

const diagrams: readonly LessonDiagram[] = lessonDiagramCatalog;

function diagramContractErrors(diagram: LessonDiagram) {
  const prefix = `第 ${diagram.lessonId} 课机制图`;
  const errors: string[] = [];
  if (diagram.nodes.length < 5 || diagram.nodes.length > 8) {
    errors.push(`${prefix}必须包含 5–8 个节点`);
  }
  if (diagram.edges.length < 5 || diagram.edges.length > 9) {
    errors.push(`${prefix}必须包含 5–9 条边`);
  }
  if (diagram.steps.length < 4 || diagram.steps.length > 6) {
    errors.push(`${prefix}必须包含 4–6 个推导步骤`);
  }
  if (diagram.facts.length < 3) {
    errors.push(`${prefix}至少需要 3 条可核对事实`);
  }

  const nodeIds = new Set(diagram.nodes.map(({ id }) => id));
  const edgeIds = new Set(diagram.edges.map(({ id }) => id));
  if (nodeIds.size !== diagram.nodes.length) {
    errors.push(`${prefix}含有重复节点 id`);
  }
  if (edgeIds.size !== diagram.edges.length) {
    errors.push(`${prefix}含有重复边 id`);
  }
  for (const edge of diagram.edges) {
    if (!nodeIds.has(edge.from) || !nodeIds.has(edge.to)) {
      errors.push(`${prefix}的边 ${edge.id} 指向不存在的节点`);
    }
  }
  const focusableIds = new Set([...nodeIds, ...edgeIds]);
  for (const [index, step] of diagram.steps.entries()) {
    if (step.focus.length === 0) {
      errors.push(`${prefix}第 ${index + 1} 步没有焦点`);
    }
    for (const focusId of step.focus) {
      if (!focusableIds.has(focusId)) {
        errors.push(
          `${prefix}第 ${index + 1} 步引用了不存在的 ${focusId}`,
        );
      }
    }
  }

  const [viewX, viewY, viewWidth, viewHeight] = (
    diagram.viewBox ?? "0 0 960 360"
  )
    .split(/\s+/)
    .map(Number);
  const bounds = diagram.nodes.map((node) => {
    const { width, height } = nodeSize(node);
    return {
      id: node.id,
      left: node.x - width / 2,
      right: node.x + width / 2,
      top: node.y - height / 2,
      bottom: node.y + height / 2,
    };
  });
  for (const node of bounds) {
    if (
      node.left < viewX + 4 ||
      node.right > viewX + viewWidth - 4 ||
      node.top < viewY + 4 ||
      node.bottom > viewY + viewHeight - 4
    ) {
      errors.push(`${prefix}的节点 ${node.id} 超出 viewBox 安全边界`);
    }
  }
  for (let leftIndex = 0; leftIndex < bounds.length; leftIndex += 1) {
    for (
      let rightIndex = leftIndex + 1;
      rightIndex < bounds.length;
      rightIndex += 1
    ) {
      const left = bounds[leftIndex];
      const right = bounds[rightIndex];
      const horizontalGap = Math.max(
        right.left - left.right,
        left.left - right.right,
      );
      const verticalGap = Math.max(
        right.top - left.bottom,
        left.top - right.bottom,
      );
      if (horizontalGap < 12 && verticalGap < 12) {
        errors.push(
          `${prefix}的节点 ${left.id} 与 ${right.id} 间距不足 12px`,
        );
      }
    }
  }
  return errors;
}

const diagramErrors = diagrams.flatMap(diagramContractErrors);
if (diagramErrors.length > 0) {
  throw new Error(diagramErrors.join("\n"));
}

export const lessonDiagramById = Object.fromEntries(
  diagrams.map((diagram) => [diagram.lessonId, diagram]),
) as Record<string, LessonDiagram>;
