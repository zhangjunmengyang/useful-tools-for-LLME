import {
  type Language,
  LANGUAGE_STORAGE_KEY,
  applyLanguageToDocument,
  persistLanguage,
  readStoredLanguage,
} from "@/lib/i18n";
import { type ReactNode, createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

interface I18nControls {
  language: Language;
  setLanguage: (language: Language) => void;
  toggleLanguage: () => void;
  tr: (zh: string, en: string) => string;
}

const I18nContext = createContext<I18nControls | null>(null);

export function useI18n(): I18nControls {
  const controls = useContext(I18nContext);
  if (!controls) throw new Error("useI18n must be used inside I18nProvider");
  return controls;
}

export function I18nProvider({ children }: { children: ReactNode }) {
  const [language, setLanguageState] = useState<Language>(readStoredLanguage);

  useEffect(() => {
    applyLanguageToDocument(language);
  }, [language]);

  useEffect(() => {
    const onStorage = (event: StorageEvent) => {
      if (event.key !== LANGUAGE_STORAGE_KEY) return;
      setLanguageState(readStoredLanguage());
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  const setLanguage = useCallback((next: Language) => {
    persistLanguage(next);
    setLanguageState(next);
  }, []);

  const toggleLanguage = useCallback(() => {
    setLanguageState((current) => {
      const next = current === "zh" ? "en" : "zh";
      persistLanguage(next);
      return next;
    });
  }, []);

  const value = useMemo<I18nControls>(
    () => ({
      language,
      setLanguage,
      toggleLanguage,
      tr: (zh, en) => (language === "zh" ? zh : en),
    }),
    [language, setLanguage, toggleLanguage],
  );

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}
