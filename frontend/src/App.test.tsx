import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it } from "vitest";

import { I18nProvider } from "./components/I18nProvider";
import { TopicSwitcher } from "./components/TopicSwitcher";
import type { TopicSummary } from "./types";

const topics: TopicSummary[] = [
  {
    id: "omni",
    title: "Omni",
    title_en: "Omni",
    short: "Omni",
    blurb: "全模态",
    blurb_en: "Omni-modal",
    kind: "local_markdown",
    ready: true,
    source: "",
    note: "",
    modes: ["read", "learn", "play"],
  },
  {
    id: "wm",
    title: "世界模型",
    title_en: "World Models",
    short: "世界模型",
    short_en: "World Models",
    blurb: "世界模型课",
    kind: "local_markdown",
    ready: true,
    source: "",
    note: "",
    modes: ["read", "learn", "play"],
  },
  {
    id: "cl",
    title: "持续学习",
    title_en: "Continual Learning",
    short: "持续学习",
    short_en: "Continual Learning",
    blurb: "持续学习课",
    kind: "local_markdown",
    ready: true,
    source: "",
    note: "",
    modes: ["read", "learn", "play"],
  },
  {
    id: "llm",
    title: "LLM",
    title_en: "LLM",
    short: "LLM",
    blurb: "大模型机制",
    kind: "local_markdown",
    ready: true,
    source: "",
    note: "",
    modes: ["read", "learn", "play"],
  },
];

function renderSwitcher() {
  return render(
    <I18nProvider>
      <MemoryRouter>
        <TopicSwitcher topics={topics} currentId="omni" />
      </MemoryRouter>
    </I18nProvider>,
  );
}

describe("TopicSwitcher", () => {
  beforeEach(() => {
    window.localStorage.setItem("app-language", "zh");
  });

  it("shows the four first-class topics", () => {
    renderSwitcher();
    expect(screen.getByRole("navigation", { name: "主题切换" })).toBeTruthy();
    expect(screen.getByRole("link", { name: /Omni/ })).toBeTruthy();
    expect(screen.getByRole("link", { name: /世界模型/ })).toBeTruthy();
    expect(screen.getByRole("link", { name: /持续学习/ })).toBeTruthy();
    expect(screen.getByRole("link", { name: /LLM/ })).toBeTruthy();
    expect(screen.getAllByRole("link")).toHaveLength(4);
  });

  it("switches topic names into English", () => {
    window.localStorage.setItem("app-language", "en");
    renderSwitcher();
    expect(screen.getByRole("navigation", { name: "Switch topic" })).toBeTruthy();
    expect(screen.getByRole("link", { name: /World Models/ })).toBeTruthy();
    expect(screen.getByRole("link", { name: /Continual Learning/ })).toBeTruthy();
  });
});
