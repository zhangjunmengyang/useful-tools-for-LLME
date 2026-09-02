import { useEffect, useState } from "react";
import { Navigate, useParams, useSearchParams } from "react-router-dom";

import { fetchLesson, fetchOutline } from "../api";
import { useI18n } from "../components/I18nProvider";
import { CurriculumRail } from "../components/CurriculumRail";
import { MarkdownBody } from "../components/MarkdownBody";
import { ToolkitIndex } from "../components/ToolkitIndex";
import { ToolWidget } from "../components/ToolWidget";
import { ReaderColumn, ReaderShell } from "../features/reader/ReaderShell";
import { localeText } from "../lib/i18n";
import type { LessonDetail, LessonMode, TopicOutline } from "../types";

export function LessonPage({
  outline,
  onOutline,
}: {
  outline: TopicOutline | null;
  onOutline: (outline: TopicOutline) => void;
}) {
  const { topicId = "", lessonId = "" } = useParams();
  const [params] = useSearchParams();
  const mode = ((params.get("mode") as LessonMode) || "read") as LessonMode;
  const { language, tr } = useI18n();
  const [lesson, setLesson] = useState<LessonDetail | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    setLesson(null);
    setError("");
    const load = async () => {
      const nextOutline = outline?.id === topicId ? outline : await fetchOutline(topicId);
      if (!cancelled && outline?.id !== nextOutline.id) onOutline(nextOutline);
      if (!lessonId) return;
      const nextLesson = await fetchLesson(topicId, lessonId);
      if (!cancelled) setLesson(nextLesson);
    };
    load().catch((err: Error) => {
      if (!cancelled) setError(err.message);
    });
    return () => {
      cancelled = true;
    };
  }, [topicId, lessonId, outline, onOutline]);

  if (!lessonId && outline?.default_lesson_id) {
    return <Navigate to={`/t/${topicId}/${outline.default_lesson_id}`} replace />;
  }

  const lessons = outline?.lessons ?? [];
  const position = Math.max(1, lessons.findIndex((item) => item.id === lessonId) + 1);
  const unit = outline?.units.find((item) => item.id === lesson?.unit_id);

  return (
    <ReaderShell
      courseTitle={localeText(language, outline?.title ?? topicId, outline?.title_en)}
      courseHref={`/t/${topicId}`}
      stageTitle={localeText(language, unit?.title ?? "", unit?.title_en) || undefined}
      chapterTitle={localeText(language, lesson?.title ?? "", lesson?.title_en) || undefined}
      position={position}
      total={Math.max(lessons.length, 1)}
      rail={
        outline ? (
          <CurriculumRail outline={outline} lessonId={lessonId} />
        ) : (
          <p className="p-4 text-sm text-muted-foreground">{tr("正在读目录…", "Loading the outline…")}</p>
        )
      }
    >
      <ReaderColumn>
        {error ? <p className="text-sm text-destructive">{error}</p> : null}
        {!lesson && !error ? (
          <p className="text-sm text-muted-foreground">{tr("正在打开这一课…", "Opening this lesson…")}</p>
        ) : null}
        {lesson ? <LessonBody lesson={lesson} outline={outline} mode={mode} /> : null}
      </ReaderColumn>
    </ReaderShell>
  );
}

function pickBody(lesson: LessonDetail, mode: LessonMode, language: "zh" | "en"): { source: string; chineseOnly: boolean } {
  const zh = mode === "learn" ? lesson.learn : mode === "play" ? lesson.play : lesson.read;
  const en = mode === "learn" ? lesson.learn_en : mode === "play" ? lesson.play_en : lesson.read_en;
  if (language === "en" && en) return { source: en, chineseOnly: false };
  return { source: zh, chineseOnly: language === "en" && lesson.body_locale !== "both" };
}

function LessonBody({
  lesson,
  outline,
  mode,
}: {
  lesson: LessonDetail;
  outline: TopicOutline | null;
  mode: LessonMode;
}) {
  const { language, tr } = useI18n();
  const { source, chineseOnly } = pickBody(lesson, mode, language);
  const body = source.replace(/^#\s+.+\n+/, "");
  const showFullToolkit = mode === "play";
  const title = localeText(language, lesson.title, lesson.title_en);
  const summary = localeText(language, lesson.summary, lesson.summary_en);
  const topicTitle = localeText(language, outline?.title ?? lesson.topic_id, outline?.title_en);

  return (
    <article className="pb-20">
      <header className="mb-6">
        <p className="text-sm text-muted-foreground">{topicTitle}</p>
        <h1 className="mt-1 text-[1.85rem] font-semibold leading-tight tracking-tight">{title}</h1>
        {summary ? <p className="mt-3 max-w-[65ch] text-[15px] leading-7 text-muted-foreground">{summary}</p> : null}
        {chineseOnly ? (
          <p className="mt-3 max-w-[65ch] text-sm text-muted-foreground">
            {tr(
              "",
              "This imported lesson is still Chinese. The chrome around it is English; the original text is kept.",
            )}
          </p>
        ) : null}
      </header>
      <MarkdownBody source={body} />
      {mode === "play" && lesson.play_tools.length > 0 ? (
        <div className="mt-8 grid gap-4">
          {lesson.play_tools.map((toolId) => (
            <ToolWidget key={toolId} toolId={toolId} />
          ))}
        </div>
      ) : null}
      {mode === "play" && lesson.original_url ? (
        <p className="mt-6 text-sm">
          <a className="text-primary-ink underline underline-offset-4" href={lesson.original_url} target="_blank" rel="noreferrer">
            {tr("打开原课（若本机已启动对应站点）", "Open the original lesson (if that site is running locally)")}
          </a>
        </p>
      ) : null}
      {showFullToolkit ? <ToolkitIndex highlightToolIds={lesson.play_tools} /> : null}
    </article>
  );
}
