"use client";

import { CheckCircle2, FlaskConical, LoaderCircle } from "lucide-react";
import {
  useEffect,
  useState,
  type ComponentType,
} from "react";
import { useProgress } from "@/components/progress/ProgressProvider";

type LabProps = {
  onComplete?: (state?: Record<string, unknown>) => void;
  initialState?: Record<string, unknown>;
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

  useEffect(() => {
    let cancelled = false;
    const numericId = Number(lessonId);
    const load =
      numericId <= 10
        ? import("@/components/labs/foundations").then(
            (module) => module.foundationLabMap[lessonId],
          )
        : import("@/components/labs/advanced").then(
            (module) => module.advancedLabMap[lessonId],
          );

    load
      .then((component) => {
        if (cancelled) return;
        if (!component) {
          setLoadState({ lessonId, Lab: null, error: true });
          return;
        }
        setLoadState({
          lessonId,
          Lab: component as ComponentType<LabProps>,
          error: false,
        });
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
            操作前先判断结果。实验只使用公开公式或确定性教学模型，真实训练结论必须回到后面的对照实验。
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
      ) : (
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
      )}
    </section>
  );
}
