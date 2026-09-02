import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson59Diagram: LessonDiagram = {
  lessonId: "59",
  title: "同一 8 路码本，两套标签",
  summary:
    "语音句和环境声可以进同一套 RVQ。语音用词序列算 WER，鼓点用毫秒 onset 算事件 F1。80 ms 帧上语音可懂，20 ms 间距的 flam 并进同一格，网格塌掉。两列不得合成一个分数。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l59-wave",
      label: ["语音句 / 鼓点"],
      meta: "同一段波形",
      kind: "input",
      x: 88,
      y: 180,
    },
    {
      id: "l59-rvq",
      label: ["8 路语音 RVQ"],
      meta: "12.5 Hz · 80 ms",
      kind: "transform",
      x: 268,
      y: 180,
    },
    {
      id: "l59-codes",
      label: ["离散码"],
      meta: "8 × 16 帧",
      kind: "state",
      x: 448,
      y: 180,
    },
    {
      id: "l59-wer",
      label: ["语音 WER"],
      meta: "词序列标签",
      kind: "decision",
      x: 648,
      y: 88,
    },
    {
      id: "l59-f1",
      label: ["事件 F1"],
      meta: "onset · τ=40 ms",
      kind: "decision",
      x: 648,
      y: 272,
    },
    {
      id: "l59-speech",
      label: ["可懂"],
      meta: "WER = 0",
      kind: "output",
      x: 848,
      y: 88,
      width: 140,
    },
    {
      id: "l59-drum",
      label: ["网格塌掉"],
      meta: "F1 = 2/3",
      kind: "output",
      x: 848,
      y: 272,
      width: 140,
    },
  ],
  edges: [
    {
      id: "l59-e-wave-rvq",
      from: "l59-wave",
      to: "l59-rvq",
      label: "24 kHz PCM",
      labelAt: { x: 178, y: 228 },
    },
    {
      id: "l59-e-rvq-codes",
      from: "l59-rvq",
      to: "l59-codes",
      label: "查 8 本字典",
      labelAt: { x: 358, y: 228 },
    },
    {
      id: "l59-e-codes-wer",
      from: "l59-codes",
      to: "l59-wer",
      label: "解码再听写",
      via: [
        { x: 530, y: 180 },
        { x: 530, y: 88 },
      ],
      labelAt: { x: 498, y: 124 },
    },
    {
      id: "l59-e-codes-f1",
      from: "l59-codes",
      to: "l59-f1",
      label: "峰值对齐",
      via: [
        { x: 530, y: 180 },
        { x: 530, y: 272 },
      ],
      labelAt: { x: 498, y: 236 },
    },
    {
      id: "l59-e-wer-out",
      from: "l59-wer",
      to: "l59-speech",
      label: "6 个词",
      labelAt: { x: 748, y: 56 },
    },
    {
      id: "l59-e-f1-out",
      from: "l59-f1",
      to: "l59-drum",
      label: "16 个 onset",
      labelAt: { x: 748, y: 304 },
    },
    {
      id: "l59-e-split",
      from: "l59-wer",
      to: "l59-f1",
      label: "不得共用分",
      labelAt: { x: 710, y: 180 },
    },
  ],
  steps: [
    {
      title: "同一段波形",
      description:
        "用户可以说「明天八点开会」，也可以敲 8 组相距 20 ms 的 flam。编码器先看到的是同一条 PCM。",
      focus: ["l59-wave", "l59-rvq", "l59-e-wave-rvq"],
    },
    {
      title: "语音 8 路网格",
      description:
        "Mimi 一类语音 codec 是 12.5 Hz、8 本 2048 行码本。名义码率约 1100 bit/s。帧长 80 ms。",
      focus: ["l59-rvq", "l59-codes", "l59-e-rvq-codes"],
    },
    {
      title: "两套标签",
      description:
        "WER 的参考是 6 个词。事件 F1 的参考是 16 个毫秒时间戳，容差 40 ms。元素类型不同，长度也不同。",
      focus: [
        "l59-codes",
        "l59-wer",
        "l59-f1",
        "l59-e-codes-wer",
        "l59-e-codes-f1",
        "l59-e-split",
      ],
    },
    {
      title: "语音可懂",
      description:
        "音节跨 1 到 2 帧。教学夹具里重建句与参考相同，WER = 0。",
      focus: ["l59-wer", "l59-speech", "l59-e-wer-out"],
    },
    {
      title: "鼓点网格塌掉",
      description:
        "20 ms 双击落到同一 80 ms 帧，16 个真 onset 只检出 8 个。Precision = 1，Recall = 1/2，F1 = 2/3。",
      focus: ["l59-f1", "l59-drum", "l59-e-f1-out"],
    },
  ],
  facts: [
    "Mimi：24 kHz、12.5 Hz、8×2048，名义码率 12.5×8×11 ≈ 1100 bit/s，帧长 80 ms。",
    "MusicGen 的 EnCodec：32 kHz、stride 640、50 Hz、4×2048；30 秒对应 1500 个 delay 步。本课不重做 delay 公式。",
    "AudioLDM-L-Full 在 AudioCaps 上 FD 23.31，走的是梅尔 VAE 连续 latent 加 CLAP，不是语音 RVQ。",
    "教学夹具：词标签 6 个，onset 标签 16 个；80 ms 网格 WER = 0、F1 = 2/3；非法共用分 (1-WER+F1)/2 = 5/6。",
    "EnCodec 原文把 speech、noisy-reverberant speech、music 分成独立 MUSHRA 域，没有给出「音频总分」。",
  ],
};
