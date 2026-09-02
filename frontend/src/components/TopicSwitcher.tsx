import { useI18n } from "@/components/I18nProvider";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { localeText } from "@/lib/i18n";
import { cn } from "@/lib/utils";
import type { TopicSummary } from "@/types";
import { Check, ChevronDown, GraduationCap } from "lucide-react";
import { NavLink, useNavigate } from "react-router-dom";

export function TopicSwitcher({
  compact = false,
  topics,
  currentId,
}: {
  compact?: boolean;
  topics: TopicSummary[];
  currentId?: string;
}) {
  const navigate = useNavigate();
  const { language, tr } = useI18n();
  const current = topics.find((topic) => topic.id === currentId) ?? topics[0];
  if (!current) return null;
  const currentTitle = localeText(language, current.title, current.title_en);

  return (
    <div>
      <nav className="sr-only" aria-label={tr("主题切换", "Switch topic")}>
        {topics.map((topic) => (
          <NavLink key={topic.id} to={`/t/${topic.id}`}>
            {localeText(language, topic.title, topic.title_en)}
          </NavLink>
        ))}
      </nav>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button
            type="button"
            aria-label={tr(`切换主题，当前${currentTitle}`, `Switch topic, now ${currentTitle}`)}
            title={compact ? tr(`当前主题：${currentTitle}`, `Current topic: ${currentTitle}`) : undefined}
            className={cn(
              "group flex h-8 items-center rounded-md text-sidebar-foreground transition-colors hover:bg-muted hover:text-sidebar-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
              compact ? "w-8 justify-center" : "w-full gap-2 px-2",
            )}
          >
            <GraduationCap className="h-3.5 w-3.5 shrink-0 text-sidebar-foreground/70" />
            {compact ? null : (
              <>
                <span className="min-w-0 flex-1 truncate text-left text-xs font-medium">{currentTitle}</span>
                <ChevronDown className="h-3.5 w-3.5 shrink-0 text-sidebar-foreground/50 transition-colors group-hover:text-sidebar-foreground" />
              </>
            )}
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" side={compact ? "right" : "bottom"} className="w-44">
          {topics.map((topic) => (
            <DropdownMenuItem key={topic.id} onSelect={() => navigate(`/t/${topic.id}`)} className="gap-2">
              <Check className={cn("h-4 w-4", current.id === topic.id ? "opacity-100" : "opacity-0")} />
              <span className="flex-1">{localeText(language, topic.title, topic.title_en)}</span>
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}
