import { LanguageSwitcher } from "@/components/LanguageSwitcher";
import { ModeTabs } from "@/components/ModeTabs";
import { useColorSchemeControls } from "@/components/ColorSchemeProvider";
import { useI18n } from "@/components/I18nProvider";
import { typographyTokens } from "@/lib/design-system";
import { localeText } from "@/lib/i18n";
import type { TopicOutline } from "@/types";
import { Moon, PanelLeft, Sun } from "lucide-react";
import { useLocation, useParams } from "react-router-dom";
import type { AppSidebarLayout } from "./useAppSidebarLayout";

export function Header({
  sidebar,
  outline,
}: {
  sidebar?: AppSidebarLayout;
  outline: TopicOutline | null;
}) {
  const { colorScheme, toggleColorScheme } = useColorSchemeControls();
  const { language, tr } = useI18n();
  const location = useLocation();
  const { lessonId } = useParams();
  const showExpand = Boolean(sidebar?.isDesktop && sidebar.collapsed);
  const showModes = Boolean(lessonId) && location.pathname.startsWith("/t/");

  let title = localeText(language, outline?.title ?? tr("学习台", "Learn Bench"), outline?.title_en);
  let description = localeText(
    language,
    outline?.blurb ?? tr("四个主题：Omni、世界模型、持续学习、LLM。", "Four topics: Omni, World Models, Continual Learning, LLM."),
    outline?.blurb_en,
  );
  if (location.pathname.startsWith("/tools")) {
    title = tr("工具台", "Tools");
    description = tr(
      "17 个 API 工具。Inspector、Run、Export、cURL 都在。",
      "All 17 API tools, with inspector, run, export, and cURL.",
    );
  } else if (location.pathname.startsWith("/explore")) {
    title = tr("实验室", "Labs");
    description = tr(
      "旧 Gradio 图表页原样挂进来：Token Arena、注意力图、LoRA、Benchmark、LLM Judge 都能开。",
      "The old Gradio chart pages, still live: Token Arena, attention maps, LoRA, Benchmark, LLM Judge.",
    );
  } else if (location.pathname.startsWith("/notebooks")) {
    title = tr("笔记本", "Notebooks");
    description = tr("额外练习，不替代工具台。", "Extras. They do not replace the tools.");
  }

  return (
    <header className="sticky top-0 z-20 flex h-20 items-center gap-3 border-b border-border bg-background px-6 md:px-8">
      {showExpand ? (
        <button
          type="button"
          onClick={sidebar?.toggleCollapsed}
          aria-label={tr("展开侧栏", "Expand sidebar")}
          aria-expanded={false}
          title={tr("展开侧栏", "Expand sidebar")}
          className="-ml-1.5 shrink-0 rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
        >
          <PanelLeft className="h-4 w-4" />
        </button>
      ) : null}
      <div className="min-w-0 flex-1">
        <h1 key={`${language}-${title}`} className={typographyTokens.pageTitle}>
          {title}
        </h1>
        <p className={`mt-1 truncate ${typographyTokens.pageDescription}`}>{description}</p>
      </div>
      {showModes ? <ModeTabs /> : null}
      <LanguageSwitcher />
      <button
        type="button"
        onClick={toggleColorScheme}
        aria-label={colorScheme === "dark" ? tr("切换到浅色", "Switch to light theme") : tr("切换到深色", "Switch to dark theme")}
        title={colorScheme === "dark" ? tr("切换到浅色", "Switch to light theme") : tr("切换到深色", "Switch to dark theme")}
        className="shrink-0 rounded-md p-2 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        {colorScheme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
      </button>
    </header>
  );
}
