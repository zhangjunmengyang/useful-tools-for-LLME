import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson43Diagram: LessonDiagram = {
  lessonId: "43",
  title: "同一句语音，两张文本 mask",
  summary:
    "音频编码器帧只当条件。ASR 交叉熵盖在听写跨度上，指令跟随交叉熵盖在执行回复上。两张有效集合不相等，转写对了仍可能指令错。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l43-wave",
      label: ["同一句语音"],
      meta: "16 kHz 波形",
      kind: "input",
      x: 88,
      y: 180,
    },
    {
      id: "l43-enc",
      label: ["音频编码器"],
      meta: "约 40 ms / 帧",
      kind: "transform",
      x: 268,
      y: 180,
    },
    {
      id: "l43-cond",
      label: ["条件序列"],
      meta: "帧 + 提示，label=-100",
      kind: "state",
      x: 458,
      y: 180,
    },
    {
      id: "l43-asr-mask",
      label: ["ASR mask"],
      meta: "M_asr = 听写跨度",
      kind: "decision",
      x: 648,
      y: 88,
    },
    {
      id: "l43-if-mask",
      label: ["指令 mask"],
      meta: "M_if = 执行跨度",
      kind: "decision",
      x: 648,
      y: 272,
    },
    {
      id: "l43-transcript",
      label: ["转写文本"],
      meta: "WER / CER",
      kind: "output",
      x: 848,
      y: 88,
      width: 150,
    },
    {
      id: "l43-action",
      label: ["执行回复"],
      meta: "16 / 7K / 声源",
      kind: "output",
      x: 848,
      y: 272,
      width: 150,
    },
  ],
  edges: [
    {
      id: "l43-e-wave-enc",
      from: "l43-wave",
      to: "l43-enc",
      label: "梅尔谱",
      labelAt: { x: 178, y: 228 },
    },
    {
      id: "l43-e-enc-cond",
      from: "l43-enc",
      to: "l43-cond",
      label: "条件帧",
      labelAt: { x: 363, y: 228 },
    },
    {
      id: "l43-e-cond-asr",
      from: "l43-cond",
      to: "l43-asr-mask",
      label: "不进损失",
      via: [
        { x: 530, y: 180 },
        { x: 530, y: 88 },
      ],
      labelAt: { x: 498, y: 124 },
    },
    {
      id: "l43-e-cond-if",
      from: "l43-cond",
      to: "l43-if-mask",
      label: "不进损失",
      via: [
        { x: 530, y: 180 },
        { x: 530, y: 272 },
      ],
      labelAt: { x: 498, y: 236 },
    },
    {
      id: "l43-e-asr-out",
      from: "l43-asr-mask",
      to: "l43-transcript",
      label: "CE 听写",
      labelAt: { x: 748, y: 56 },
    },
    {
      id: "l43-e-if-out",
      from: "l43-if-mask",
      to: "l43-action",
      label: "CE 执行",
      labelAt: { x: 748, y: 304 },
    },
    {
      id: "l43-e-asr-if",
      from: "l43-asr-mask",
      to: "l43-if-mask",
      label: "集合不相等",
      labelAt: { x: 710, y: 180 },
    },
  ],
  steps: [
    {
      title: "同一段波形",
      description:
        "用户说的那句话既是可听写的文本，也是可执行的命令。编码器只负责把 16 kHz 波形变成帧。",
      focus: ["l43-wave", "l43-enc", "l43-e-wave-enc"],
    },
    {
      title: "帧是条件",
      description:
        "Qwen2-Audio 每帧约 40 ms。这些位置的 label 为 -100，不进入文本交叉熵。",
      focus: ["l43-enc", "l43-cond", "l43-e-enc-cond"],
    },
    {
      title: "两张 mask",
      description:
        "ASR 有效集合盖在听写跨度，指令有效集合盖在执行回复。教学序列上分别是 5 个与 3 个 token。",
      focus: [
        "l43-cond",
        "l43-asr-mask",
        "l43-if-mask",
        "l43-e-cond-asr",
        "l43-e-cond-if",
        "l43-e-asr-if",
      ],
    },
    {
      title: "两条交叉熵",
      description:
        "复读模型在听写位置概率 0.80、执行位置 0.15。换一张 mask，损失可以差一个数量级。",
      focus: [
        "l43-asr-mask",
        "l43-if-mask",
        "l43-transcript",
        "l43-action",
        "l43-e-asr-out",
        "l43-e-if-out",
      ],
    },
    {
      title: "转写对仍可能指令错",
      description:
        "SALMONN 无激活时 test-clean WER 已是 2.1%，讲故事跟随率是 0.00。Qwen2-Audio Librispeech 1.6%，口述 MMLU 只有 33.2。",
      focus: ["l43-transcript", "l43-action", "l43-asr-mask", "l43-if-mask"],
    },
  ],
  facts: [
    "Qwen2-Audio：16 kHz、128 通道梅尔、25 ms 窗 / 10 ms hop、池化步长 2，约 40 ms/帧；总参数 8.2B。",
    "SALMONN：Whisper-Large-v2 + BEATs，窗口 Q-Former N=1、L=17，约 88 个文本侧 token / 30 秒；激活后 Story FR 从 0.00 到 1.00，LibriSpeech test-clean WER 仍为 2.1%。",
    "Qwen2.5-Omni 表 4：约 90% 可口述子集上，口述 MMLU 65.6、GSM8K 85.4；Qwen2-Audio 同表为 33.2 与 18.4。",
    "教学夹具 M_asr={11..15}，M_if={16,17,18}，两集合不相等且不相交；音频与提示位置 label=-100。",
    "Qwen2-Audio AIR-Bench chat 四维 7.18 / 6.99 / 6.79 / 6.77，测的是指令层，不是 WER。",
  ],
};
