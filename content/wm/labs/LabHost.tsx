"use client";

import { CheckCircle2, FlaskConical, LoaderCircle } from "lucide-react";
import { useEffect, useState, type ComponentType } from "react";
import { useProgress } from "@/components/progress/ProgressProvider";

type LabProps = {
  onComplete?: (state?: Record<string, unknown>) => void;
  initialState?: Record<string, unknown>;
};

const labRegistry: Record<
  string,
  () => Promise<ComponentType<LabProps> | undefined>
> = {
  "03": () => import("@/components/labs/ActionSwapLab").then((mod) => mod.default),
  "16": () => import("@/components/labs/ThreeRoadsLab").then((mod) => mod.default),
  "21": () => import("@/components/labs/PermanenceLab").then((mod) => mod.default),
  "23": () => import("@/components/labs/SlotLab").then((mod) => mod.default),
  "27": () => import("@/components/labs/SimGapLab").then((mod) => mod.default),
  "30": () => import("@/components/labs/ActionSwapLab").then((mod) => mod.default),
  "31": () => import("@/components/labs/GazeLab").then((mod) => mod.default),
  "32": () => import("@/components/labs/DeskPetLab").then((mod) => mod.default),
  "33": () => import("@/components/labs/EmbodimentLab").then((mod) => mod.default),
  "34": () => import("@/components/labs/LeaderboardLab").then((mod) => mod.default),
  "35": () => import("@/components/labs/OmnimodalLab").then((mod) => mod.default),
  "36": () => import("@/components/labs/AudioWorldLab").then((mod) => mod.default),
  "37": () => import("@/components/labs/RecipeLab").then((mod) => mod.default),
  "38": () => import("@/components/labs/ArchZooLab").then((mod) => mod.default),
  "39": () => import("@/components/labs/MemoryHorizonLab").then((mod) => mod.default),
  "40": () => import("@/components/labs/PipelineLab").then((mod) => mod.default),
  "41": () => import("@/components/labs/DrivingCondLab").then((mod) => mod.default),
  "42": () => import("@/components/labs/ThreeHeadsLab").then((mod) => mod.default),
  "43": () => import("@/components/labs/PlayableLab").then((mod) => mod.default),
  "44": () => import("@/components/labs/PhysicsScoreLab").then((mod) => mod.default),
  "45": () => import("@/components/labs/ObjectiveLab").then((mod) => mod.default),
};

export function LabHost({ lessonId }: { lessonId: string }) {
  const [loadState, setLoadState] = useState<{
    lessonId: string;
    Lab: ComponentType<LabProps> | null;
    error: boolean;
  }>({ lessonId: "", Lab: null, error: false });
  const { state, saveLabState } = useProgress();
  const lessonProgress = state.lessons[lessonId];
  const currentLoad =
    loadState.lessonId === lessonId
      ? loadState
      : { lessonId, Lab: null, error: false };
  const ActiveLab = currentLoad.Lab;
  const registered = Boolean(labRegistry[lessonId]);

  useEffect(() => {
    const load = labRegistry[lessonId];
    if (!load) return;
    let cancelled = false;
    load()
      .then((component) => {
        if (cancelled) return;
        if (!component) {
          setLoadState({ lessonId, Lab: null, error: true });
          return;
        }
        setLoadState({ lessonId, Lab: component, error: false });
      })
      .catch(() => {
        if (!cancelled) {
          setLoadState({ lessonId, Lab: null, error: true });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [lessonId]);

  return (
    <section
      id="interactive-lab"
      className="interactive-lab-shell"
      aria-labelledby={`interactive-lab-heading-${lessonId}`}
    >
      <div className="interactive-lab-heading">
        <div>
          <span className="evidence-label simulation">
            <FlaskConical aria-hidden="true" size={15} />
            交互实验
          </span>
          <h2 id={`interactive-lab-heading-${lessonId}`}>
            先预测，再运行实验
          </h2>
          <p>
            操作前先判断结果。页面上的交互实验用于理解机制；真实训练结论要回到锚定仓库的正式实验。
          </p>
        </div>
        {lessonProgress?.labCompleted ? (
          <span className="lab-complete-state">
            <CheckCircle2 aria-hidden="true" size={17} />
            本课实验已验收
          </span>
        ) : null}
      </div>

      {ActiveLab ? (
        <ActiveLab
          initialState={lessonProgress?.labState}
          onComplete={(labState) => saveLabState(lessonId, labState)}
        />
      ) : registered ? (
        <div className="lab-loading" role="status">
          {currentLoad.error ? (
            <p>这项交互实验没有加载成功，请刷新页面再试。</p>
          ) : (
            <>
              <LoaderCircle className="spin" aria-hidden="true" size={21} />
              <span>正在载入本课实验…</span>
            </>
          )}
        </div>
      ) : (
        <div className="lab-loading" role="status">
          <p>本课的交互实验组件随课程正文一起交付，正文发布后在这里上线。</p>
        </div>
      )}
    </section>
  );
}
