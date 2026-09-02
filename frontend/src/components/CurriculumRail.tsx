import { useI18n } from "@/components/I18nProvider";
import { localeText } from "@/lib/i18n";
import { cn } from "@/lib/utils";
import type { TopicOutline } from "@/types";
import { NavLink, useSearchParams } from "react-router-dom";

const SUBITEM_CLASS =
  "flex min-w-0 items-center gap-1.5 border-l py-1.5 pl-3 pr-3 text-xs leading-5 transition-colors";

export function CurriculumRail({ outline, lessonId }: { outline: TopicOutline; lessonId?: string }) {
  const [params] = useSearchParams();
  const mode = params.get("mode") ?? "read";
  const search = mode === "read" ? "" : `?mode=${mode}`;
  const { language, tr } = useI18n();

  return (
    <nav className="h-full overflow-y-auto px-0 pb-8 pt-3" aria-label={tr("课程目录", "Lesson list")}>
      {outline.units.map((unit) => (
        <section key={unit.id} className="mb-4">
          <p className="px-3 pb-1.5 text-2xs font-medium text-muted-foreground">
            {localeText(language, unit.title, unit.title_en)}
          </p>
          <ol className="border-l border-border">
            {unit.lessons.map((lesson, index) => {
              const active = lesson.id === lessonId;
              const title = localeText(language, lesson.title, lesson.title_en);
              return (
                <li key={lesson.id} className="-ml-px">
                  <NavLink
                    to={`/t/${outline.id}/${lesson.id}${search}`}
                    aria-current={active ? "page" : undefined}
                    className={cn(
                      SUBITEM_CLASS,
                      active
                        ? "border-l-2 border-primary bg-primary-muted/50 font-medium text-primary-ink"
                        : "border-transparent text-muted-foreground hover:bg-muted/50 hover:text-foreground",
                    )}
                    title={title}
                  >
                    <span
                      className={cn(
                        "min-w-9 shrink-0 whitespace-nowrap text-right font-mono text-2xs tabular-nums",
                        active ? "text-primary-ink" : "text-muted-foreground/60",
                      )}
                    >
                      {lesson.number ?? String(index + 1).padStart(2, "0")}
                    </span>
                    <span className="min-w-0 flex-1 truncate">{title}</span>
                  </NavLink>
                </li>
              );
            })}
          </ol>
        </section>
      ))}
    </nav>
  );
}
