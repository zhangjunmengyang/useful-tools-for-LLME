import type { DiagramNode } from "@/lib/lesson-diagrams/types";

export const DEFAULT_NODE_WIDTH = 148;
export const DEFAULT_NODE_HEIGHT = 62;

function glyphWidth(character: string, fontSize: number) {
  if (/\s/.test(character)) return fontSize * 0.34;
  if (/[\u0000-\u00ff]/.test(character)) {
    if (/[A-Z0-9]/.test(character)) return fontSize * 0.62;
    if (/[ilI1|.,:;'`]/.test(character)) return fontSize * 0.3;
    return fontSize * 0.54;
  }
  return fontSize;
}

export function measureSvgText(text: string, fontSize: number) {
  return Array.from(text).reduce(
    (width, character) => width + glyphWidth(character, fontSize),
    0,
  );
}

function preferredBreakIndex(text: string) {
  return Math.max(
    text.lastIndexOf(" "),
    text.lastIndexOf("/"),
    text.lastIndexOf("·"),
    text.lastIndexOf("+"),
    text.lastIndexOf("×"),
  );
}

export function wrapSvgText(
  text: string,
  maxWidth: number,
  fontSize: number,
) {
  if (measureSvgText(text, fontSize) <= maxWidth) return [text];

  const lines: string[] = [];
  let current = "";
  for (const character of Array.from(text)) {
    const candidate = current + character;
    if (current && measureSvgText(candidate, fontSize) > maxWidth) {
      const breakAt = preferredBreakIndex(current);
      if (breakAt > 0) {
        const includesDelimiter = current[breakAt] !== " ";
        lines.push(
          current.slice(0, breakAt + (includesDelimiter ? 1 : 0)).trim(),
        );
        current =
          current.slice(breakAt + 1).trimStart() + character;
      } else {
        lines.push(current);
        current = character;
      }
    } else {
      current = candidate;
    }
  }
  if (current.trim()) lines.push(current.trim());
  return lines;
}

export function nodeTextLayout(node: DiagramNode) {
  const width = node.width ?? DEFAULT_NODE_WIDTH;
  const labelLines = node.label.flatMap((line) =>
    wrapSvgText(line, width - 24, 12),
  );
  const metaLines = node.meta
    ? wrapSvgText(node.meta, width - 20, 10)
    : [];
  return { labelLines, metaLines };
}

export function nodeSize(node: DiagramNode) {
  const width = node.width ?? DEFAULT_NODE_WIDTH;
  const { labelLines, metaLines } = nodeTextLayout(node);
  const labelHeight = Math.max(1, labelLines.length) * 16;
  const metaHeight = metaLines.length * 13;
  const contentHeight =
    labelHeight + (metaLines.length > 0 ? 4 + metaHeight : 0);
  return {
    width,
    height: Math.max(
      node.height ?? DEFAULT_NODE_HEIGHT,
      contentHeight + 16,
    ),
  };
}
