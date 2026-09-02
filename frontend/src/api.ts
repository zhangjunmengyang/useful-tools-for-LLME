import type { LabsPayload, LessonDetail, TopicOutline, TopicSummary, ToolRun, ToolsPayload } from "./types";

function apiFail(zh: string, en: string): string {
  if (typeof document !== "undefined" && document.documentElement.lang.startsWith("en")) return en;
  return zh;
}

async function readJson<T>(response: Response, fallback: string): Promise<T> {
  if (!response.ok) {
    throw new Error(`${fallback}: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function fetchTopics(): Promise<TopicSummary[]> {
  const payload = await readJson<{ topics: TopicSummary[] }>(
    await fetch("/api/learn/topics"),
    apiFail("无法读取主题", "Could not load topics"),
  );
  return payload.topics;
}

export async function fetchOutline(topicId: string): Promise<TopicOutline> {
  return readJson<TopicOutline>(
    await fetch(`/api/learn/topics/${topicId}`),
    apiFail("无法读取大纲", "Could not load the outline"),
  );
}

export async function fetchLesson(topicId: string, lessonId: string): Promise<LessonDetail> {
  return readJson<LessonDetail>(
    await fetch(`/api/learn/topics/${topicId}/lessons/${lessonId}`),
    apiFail("无法读取课文", "Could not load the lesson"),
  );
}

export async function fetchTools(): Promise<ToolsPayload> {
  return readJson<ToolsPayload>(await fetch("/api/tools"), apiFail("无法读取工具", "Could not load tools"));
}

export async function runTool(toolId: string, inputs: Record<string, unknown>): Promise<ToolRun> {
  return readJson<ToolRun>(
    await fetch(`/api/tools/${toolId}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(inputs),
    }),
    apiFail("工具没有跑起来", "The tool did not run"),
  );
}

export async function exportTool(toolId: string, inputs: Record<string, unknown>): Promise<ToolRun> {
  return readJson<ToolRun>(
    await fetch(`/api/tools/${toolId}/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ inputs }),
    }),
    apiFail("工具导出失败", "Tool export failed"),
  );
}

export async function fetchLabs(): Promise<LabsPayload> {
  return readJson<LabsPayload>(await fetch("/api/labs"), apiFail("无法读取实验室", "Could not load labs"));
}
