import { List, X } from "@phosphor-icons/react";
import { useState } from "react";
import { Outlet } from "react-router-dom";

import { cn } from "../lib/cn";
import type { TopicOutline, TopicSummary } from "../types";
import { CurriculumRail } from "./CurriculumRail";
import { ModeTabs } from "./ModeTabs";
import { TopicSwitcher } from "./TopicSwitcher";

export function Shell({
  topics,
  outline,
  lessonId,
}: {
  topics: TopicSummary[];
  outline: TopicOutline | null;
  lessonId?: string;
}) {
  const [railOpen, setRailOpen] = useState(false);

  return (
    <div className="flex h-dvh overflow-hidden bg-canvas text-ink">
      <aside
        className={cn(
          "flex w-[min(20rem,86vw)] shrink-0 flex-col border-r border-line bg-chrome",
          "max-md:fixed max-md:inset-y-0 max-md:left-0 max-md:z-30 max-md:shadow-lg",
          railOpen ? "max-md:flex" : "max-md:hidden",
        )}
      >
        <div className="flex items-center justify-between px-3 pb-1 pt-3">
          <p className="text-[15px] font-semibold tracking-tight">学习台</p>
          <button type="button" className="p-1 text-mute md:hidden" onClick={() => setRailOpen(false)} aria-label="关闭目录">
            <X className="h-5 w-5" weight="bold" />
          </button>
        </div>
        <div className="px-3 pt-2">
          <TopicSwitcher topics={topics} currentId={outline?.id} />
        </div>
        {outline ? <CurriculumRail outline={outline} lessonId={lessonId} /> : <p className="p-4 text-sm text-mute">正在读目录…</p>}
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 shrink-0 items-center justify-between gap-3 border-b border-line bg-chrome px-3 md:px-6">
          <button type="button" className="p-1 text-ink md:hidden" onClick={() => setRailOpen(true)} aria-label="打开目录">
            <List className="h-5 w-5" weight="bold" />
          </button>
          <p className="min-w-0 truncate text-sm text-mute">{outline?.blurb ?? "四个主题：Omni、世界模型、持续学习、LLM。"}</p>
          <ModeTabs />
        </header>
        <main className="min-h-0 flex-1 overflow-y-auto bg-paper px-4 py-6 md:px-10 md:py-8">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
