import { foundationDiagrams } from "@/lib/lesson-diagrams/foundations";
import { nodeSize } from "@/lib/lesson-diagrams/layout";
import { systemDiagrams } from "@/lib/lesson-diagrams/systems";
import { trainingDiagrams } from "@/lib/lesson-diagrams/training";
import type { LessonDiagram } from "@/lib/lesson-diagrams/types";
import { lesson21Diagram } from "@/lib/lesson-diagrams/expansion/lesson_21";
import { lesson22Diagram } from "@/lib/lesson-diagrams/expansion/lesson_22";
import { lesson23Diagram } from "@/lib/lesson-diagrams/expansion/lesson_23";
import { lesson24Diagram } from "@/lib/lesson-diagrams/expansion/lesson_24";
import { lesson25Diagram } from "@/lib/lesson-diagrams/expansion/lesson_25";
import { lesson26Diagram } from "@/lib/lesson-diagrams/expansion/lesson_26";
import { lesson27Diagram } from "@/lib/lesson-diagrams/expansion/lesson_27";
import { lesson28Diagram } from "@/lib/lesson-diagrams/expansion/lesson_28";
import { lesson29Diagram } from "@/lib/lesson-diagrams/expansion/lesson_29";
import { lesson30Diagram } from "@/lib/lesson-diagrams/expansion/lesson_30";
import { lesson31Diagram } from "@/lib/lesson-diagrams/expansion/lesson_31";
import { lesson32Diagram } from "@/lib/lesson-diagrams/expansion/lesson_32";
import { lesson33Diagram } from "@/lib/lesson-diagrams/expansion/lesson_33";
import { lesson34Diagram } from "@/lib/lesson-diagrams/expansion/lesson_34";
import { lesson35Diagram } from "@/lib/lesson-diagrams/expansion/lesson_35";
import { lesson36Diagram } from "@/lib/lesson-diagrams/expansion/lesson_36";
import { lesson37Diagram } from "@/lib/lesson-diagrams/expansion/lesson_37";
import { lesson38Diagram } from "@/lib/lesson-diagrams/expansion/lesson_38";
import { lesson39Diagram } from "@/lib/lesson-diagrams/expansion/lesson_39";
import { lesson40Diagram } from "@/lib/lesson-diagrams/expansion/lesson_40";
import { lesson41Diagram } from "@/lib/lesson-diagrams/expansion/lesson_41";
import { lesson42Diagram } from "@/lib/lesson-diagrams/expansion/lesson_42";
import { lesson43Diagram } from "@/lib/lesson-diagrams/expansion/lesson_43";
import { lesson44Diagram } from "@/lib/lesson-diagrams/expansion/lesson_44";
import { lesson45Diagram } from "@/lib/lesson-diagrams/expansion/lesson_45";
import { lesson46Diagram } from "@/lib/lesson-diagrams/expansion/lesson_46";
import { lesson47Diagram } from "@/lib/lesson-diagrams/expansion/lesson_47";
import { lesson48Diagram } from "@/lib/lesson-diagrams/expansion/lesson_48";
import { lesson49Diagram } from "@/lib/lesson-diagrams/expansion/lesson_49";
import { lesson50Diagram } from "@/lib/lesson-diagrams/expansion/lesson_50";
import { lesson51Diagram } from "@/lib/lesson-diagrams/expansion/lesson_51";
import { lesson52Diagram } from "@/lib/lesson-diagrams/expansion/lesson_52";
import { lesson53Diagram } from "@/lib/lesson-diagrams/expansion/lesson_53";
import { lesson54Diagram } from "@/lib/lesson-diagrams/expansion/lesson_54";
import { lesson55Diagram } from "@/lib/lesson-diagrams/expansion/lesson_55";
import { lesson56Diagram } from "@/lib/lesson-diagrams/expansion/lesson_56";
import { lesson57Diagram } from "@/lib/lesson-diagrams/expansion/lesson_57";
import { lesson58Diagram } from "@/lib/lesson-diagrams/expansion/lesson_58";
import { lesson59Diagram } from "@/lib/lesson-diagrams/expansion/lesson_59";
import { lesson60Diagram } from "@/lib/lesson-diagrams/expansion/lesson_60";

const diagrams = [
  ...foundationDiagrams,
  ...systemDiagrams,
  ...trainingDiagrams,
  lesson21Diagram,
  lesson22Diagram,
  lesson23Diagram,
  lesson24Diagram,
  lesson25Diagram,
  lesson26Diagram,
  lesson27Diagram,
  lesson28Diagram,
  lesson29Diagram,
  lesson30Diagram,
  lesson31Diagram,
  lesson32Diagram,
  lesson33Diagram,
  lesson34Diagram,
  lesson35Diagram,
  lesson36Diagram,
  lesson37Diagram,
  lesson38Diagram,
  lesson39Diagram,
  lesson40Diagram,
  lesson41Diagram,
  lesson42Diagram,
  lesson43Diagram,
  lesson44Diagram,
  lesson45Diagram,
  lesson46Diagram,
  lesson47Diagram,
  lesson48Diagram,
  lesson49Diagram,
  lesson50Diagram,
  lesson51Diagram,
  lesson52Diagram,
  lesson53Diagram,
  lesson54Diagram,
  lesson55Diagram,
  lesson56Diagram,
  lesson57Diagram,
  lesson58Diagram,
  lesson59Diagram,
  lesson60Diagram,
] satisfies readonly LessonDiagram[];

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

const expectedLessonIds = Array.from(
  { length: 60 },
  (_, index) => String(index + 1).padStart(2, "0"),
);
if (
  diagrams.length !== expectedLessonIds.length ||
  diagrams.some(
    (diagram, index) => diagram.lessonId !== expectedLessonIds[index],
  )
) {
  throw new Error("60 课机制图必须按 01–60 完整注册");
}
const diagramErrors = diagrams.flatMap(diagramContractErrors);
if (diagramErrors.length > 0) {
  throw new Error(diagramErrors.join("\n"));
}

export const lessonDiagramById = Object.fromEntries(
  diagrams.map((diagram) => [diagram.lessonId, diagram]),
) as Record<string, LessonDiagram>;
