"use client";

import { useState } from "react";
import { LabShell } from "@/components/labs/LabShell";

const behaviors = [
  {
    id: "permanence",
    title: "物体恒常",
    query: "杯子被手挡住，状态里杯子仍在桌左。",
    future: "转头看向估计位置，杯子应还在。",
    safety: "不移动手臂。",
    result: "头转向杯位。没有查询状态的转头不算。",
  },
  {
    id: "gaze",
    title: "对视",
    query: "人下一秒看镜头的概率 0.76。",
    future: "对视后打扰成本低。",
    safety: "人若低头则不要凑近。",
    result: "看向人脸。概率来自第 31 课的头，不是聊天模型。",
  },
  {
    id: "hold",
    title: "克制",
    query: "手到杯的距离 8 cm，桌沿 4 cm。",
    future: "伸手轨迹在 0.6 秒后过桌沿。",
    safety: "截断伸手，改成提示。",
    result: "没有执行该动作。纯规则 if 距离小就停，若没查询世界模型，记零。",
  },
  {
    id: "swap",
    title: "动作对换",
    query: "同一历史帧。",
    future: "看左：头左转，杯不动。伸手：杯向桌沿。",
    safety: "只展示，不真的伸手。",
    result: "两条想象并排。分岔失败则毕业不及格。",
  },
  {
    id: "abstain",
    title: "失败承认",
    query: "世界模型熵高，最近没见过这种摆放。",
    future: "多条未来互相矛盾。",
    safety: "停下。",
    result: "说不知道。乱动比停更糟。",
  },
] as const;

export default function DeskPetLab({
  onComplete,
}: {
  onComplete?: (state?: Record<string, unknown>) => void;
}) {
  const [current, setCurrent] = useState<(typeof behaviors)[number]["id"]>("hold");
  const [seen, setSeen] = useState<string[]>(["hold"]);
  const behavior = behaviors.find((item) => item.id === current) ?? behaviors[2];

  return (
    <LabShell
      brief="毕业控制台。五件行为每件都必须先查询世界模型。点开看查询、想象、安全层和真实结果。"
      verdict={
        seen.length >= 5
          ? "五件都点过了。回到第 32 课用自己的日志证明每一件都调用了 wm_forward。"
          : `已看 ${seen.length} / 5 件。没有查询的行为不能算毕业。`
      }
    >
      <div className="wm-lab-toolbar">
        {behaviors.map((item) => (
          <button
            key={item.id}
            type="button"
            aria-pressed={current === item.id}
            onClick={() => {
              setCurrent(item.id);
              setSeen((list) => (list.includes(item.id) ? list : [...list, item.id]));
            }}
          >
            {item.title}
          </button>
        ))}
        <button
          type="button"
          onClick={() => onComplete?.({ current, seen })}
        >
          记下控制台
        </button>
      </div>
      <ul className="wm-lab-list">
        <li>
          <strong>查询的状态</strong>
          {behavior.query}
        </li>
        <li>
          <strong>展开的未来</strong>
          {behavior.future}
        </li>
        <li>
          <strong>安全层改了什么</strong>
          {behavior.safety}
        </li>
        <li>
          <strong>真实结果</strong>
          {behavior.result}
        </li>
      </ul>
    </LabShell>
  );
}
