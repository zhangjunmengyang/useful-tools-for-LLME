import { useI18n } from "@/components/I18nProvider";
import { cn } from "@/lib/utils";

export function LanguageSwitcher() {
  const { language, setLanguage, tr } = useI18n();

  return (
    <fieldset className="inline-flex shrink-0 items-center rounded-md border border-border bg-muted p-0.5">
      <legend className="sr-only">{tr("语言", "Language")}</legend>
      {(["zh", "en"] as const).map((item) => (
        <button
          key={item}
          type="button"
          aria-pressed={language === item}
          aria-label={item === "zh" ? tr("中文", "Chinese") : tr("英文", "English")}
          onClick={() => setLanguage(item)}
          className={cn(
            "h-7 rounded-sm px-2 text-xs font-medium transition-colors",
            language === item
              ? "bg-card text-foreground shadow-depth-1"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          {item === "zh" ? "中" : "EN"}
        </button>
      ))}
    </fieldset>
  );
}
