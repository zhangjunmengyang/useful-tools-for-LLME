import { useCallback, useEffect, useState } from "react";
import { Navigate, Outlet, RouterProvider, createBrowserRouter, useParams } from "react-router-dom";

import { fetchOutline, fetchTopics } from "./api";
import { useI18n } from "./components/I18nProvider";
import { MainLayout } from "./components/layout/MainLayout";
import { LabsExplorer } from "./pages/LabsExplorer";
import { LessonPage } from "./pages/LessonPage";
import { NotebooksPage } from "./pages/NotebooksPage";
import { ToolsPage } from "./pages/ToolsPage";
import type { TopicOutline, TopicSummary } from "./types";

function AppFrame() {
  const { topicId } = useParams();
  const { tr } = useI18n();
  const [topics, setTopics] = useState<TopicSummary[]>([]);
  const [outline, setOutline] = useState<TopicOutline | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchTopics()
      .then(setTopics)
      .catch((err: Error) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!topicId) return;
    let cancelled = false;
    fetchOutline(topicId)
      .then((next) => {
        if (!cancelled) setOutline(next);
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [topicId]);

  if (error && topics.length === 0) {
    return (
      <div className="grid h-dvh place-items-center bg-background px-6 text-foreground">
        <p className="max-w-md text-sm">
          {error}. {tr("先启动 API：python scripts/dev_workbench.py", "Start the API first: python scripts/dev_workbench.py")}
        </p>
      </div>
    );
  }

  return (
    <MainLayout topics={topics} outline={outline}>
      <Outlet />
    </MainLayout>
  );
}

function TopicIndex() {
  const { topicId = "" } = useParams();
  const { tr } = useI18n();
  const [target, setTarget] = useState<string | null>(null);
  useEffect(() => {
    fetchOutline(topicId)
      .then((next) => setTarget(next.default_lesson_id))
      .catch(() => setTarget(""));
  }, [topicId]);
  if (target === null) return <p className="text-sm text-muted-foreground">{tr("正在打开这个主题…", "Opening this topic…")}</p>;
  if (!target) return <p className="text-sm text-muted-foreground">{tr("这个主题还没有课。", "This topic has no lessons yet.")}</p>;
  return <Navigate to={`/t/${topicId}/${target}`} replace />;
}

function RootRedirect() {
  return <Navigate to="/t/omni" replace />;
}

function LessonRoute() {
  const { topicId } = useParams();
  const [outline, setOutline] = useState<TopicOutline | null>(null);
  const onOutline = useCallback((next: TopicOutline) => setOutline(next), []);
  useEffect(() => {
    if (!topicId) return;
    fetchOutline(topicId).then(setOutline).catch(() => undefined);
  }, [topicId]);
  return <LessonPage outline={outline} onOutline={onOutline} />;
}

const router = createBrowserRouter([
  {
    path: "/",
    element: <AppFrame />,
    children: [
      { index: true, element: <RootRedirect /> },
      { path: "tools", element: <ToolsPage />, handle: { fill: true, wide: true } },
      { path: "explore", element: <LabsExplorer />, handle: { fill: true, wide: true } },
      { path: "explore/:pageId", element: <LabsExplorer />, handle: { fill: true, wide: true } },
      { path: "notebooks", element: <NotebooksPage /> },
      { path: "t/:topicId", element: <TopicIndex /> },
      { path: "t/:topicId/:lessonId", element: <LessonRoute />, handle: { fill: true } },
    ],
  },
]);

export default function App() {
  return <RouterProvider router={router} />;
}
