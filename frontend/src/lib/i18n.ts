export type Language = "zh" | "en";

export const LANGUAGE_STORAGE_KEY = "app-language";

function browserLanguage(): Language {
  if (typeof navigator === "undefined") return "zh";
  const raw = (navigator.language || "").toLowerCase();
  return raw.startsWith("en") ? "en" : "zh";
}

export function readStoredLanguage(): Language {
  if (typeof window === "undefined") return "zh";
  try {
    const raw = window.localStorage.getItem(LANGUAGE_STORAGE_KEY);
    if (raw === "en" || raw === "zh") return raw;
  } catch {
    // Privacy mode: fall through to the browser language.
  }
  return browserLanguage();
}

export function persistLanguage(language: Language): void {
  try {
    window.localStorage.setItem(LANGUAGE_STORAGE_KEY, language);
  } catch {
    // Keep the in-memory choice for this session.
  }
}

export function applyLanguageToDocument(language: Language): void {
  document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
  document.title = language === "zh" ? "学习台" : "Learn Bench";
}

export function localeText(language: Language, zh: string, en?: string | null): string {
  if (language === "en" && en) return en;
  return zh;
}
