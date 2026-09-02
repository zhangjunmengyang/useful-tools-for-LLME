import { useI18n } from "@/components/I18nProvider";
import { cn } from "@/lib/utils";
import type { LessonMode } from "@/types";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";

export function ModeTabs() {
  const { topicId = "", lessonId = "" } = useParams();
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const mode = (params.get("mode") as LessonMode) || "read";
  const { tr } = useI18n();
  const modes: { id: LessonMode; label: string }[] = [
    { id: "read", label: tr("读", "Read") },
    { id: "learn", label: tr("学", "Learn") },
    { id: "play", label: tr("玩", "Play") },
  ];

  if (!lessonId) return null;

  return (
    <fieldset className="flex items-center">
      <legend className="sr-only">{tr("读、学、玩", "Read, Learn, Play")}</legend>
      {modes.map((item) => {
        const selected = item.id === mode;
        return (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-selected={selected}
            aria-label={item.label}
            onClick={() => {
              const next = new URLSearchParams(params);
              if (item.id === "read") next.delete("mode");
              else next.set("mode", item.id);
              const query = next.toString();
              navigate(`/t/${topicId}/${lessonId}${query ? `?${query}` : ""}`);
            }}
            className={cn(
              "rounded-md px-2.5 py-1 text-xs font-medium transition-colors",
              selected
                ? "bg-muted text-foreground"
                : "text-muted-foreground hover:bg-muted hover:text-foreground",
            )}
          >
            {item.label}
          </button>
        );
      })}
    </fieldset>
  );
}
