import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson48Diagram: LessonDiagram = {
  lessonId: "48",
  title: "一张状态表上的两列时钟",
  summary:
    "语音事件和手臂事件写进同一行状态，却必须分列记录 audio_available_at 与 action_available_at。PAUSE 只停未播出队；REPLAN 取消未播 PCM 和未执行剩余步，不得撤回已播音频或已发生接触。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l48-in",
      label: ["语音与杯子"],
      meta: "用户插话 / 挪杯",
      kind: "input",
      x: 96,
      y: 86,
      width: 136,
    },
    {
      id: "l48-audio",
      label: ["音频时钟"],
      meta: "块 320 ms",
      kind: "state",
      x: 308,
      y: 86,
      width: 132,
    },
    {
      id: "l48-action",
      label: ["控制时钟"],
      meta: "周期 1/f",
      kind: "state",
      x: 518,
      y: 86,
      width: 132,
    },
    {
      id: "l48-stale",
      label: ["过期判定"],
      meta: "帧 vs H/f",
      kind: "decision",
      x: 742,
      y: 86,
      width: 132,
    },
    {
      id: "l48-row",
      label: ["一行状态表"],
      meta: "两列时间戳",
      kind: "state",
      x: 200,
      y: 254,
      width: 148,
    },
    {
      id: "l48-ctrl",
      label: ["控制头"],
      meta: "PAUSE / REPLAN",
      kind: "decision",
      x: 470,
      y: 254,
      width: 156,
    },
    {
      id: "l48-out",
      label: ["未来队列"],
      meta: "未播 PCM / 剩余步",
      kind: "output",
      x: 742,
      y: 254,
      width: 148,
    },
  ],
  edges: [
    {
      id: "l48-e-in-audio",
      from: "l48-in",
      to: "l48-audio",
      label: "codec 块",
      labelAt: { x: 198, y: 54 },
    },
    {
      id: "l48-e-in-action",
      from: "l48-in",
      to: "l48-action",
      label: "观察锁存",
      via: [
        { x: 96, y: 36 },
        { x: 518, y: 36 },
      ],
      labelAt: { x: 300, y: 20 },
    },
    {
      id: "l48-e-audio-stale",
      from: "l48-audio",
      to: "l48-stale",
      label: "audio_available_at",
      labelAt: { x: 530, y: 54 },
    },
    {
      id: "l48-e-action-stale",
      from: "l48-action",
      to: "l48-stale",
      label: "action_available_at",
    },
    {
      id: "l48-e-audio-row",
      from: "l48-audio",
      to: "l48-row",
      label: "写入音频列",
    },
    {
      id: "l48-e-action-row",
      from: "l48-action",
      to: "l48-row",
      label: "写入动作列",
      via: [
        { x: 518, y: 170 },
        { x: 200, y: 170 },
      ],
      labelAt: { x: 360, y: 154 },
    },
    {
      id: "l48-e-stale-ctrl",
      from: "l48-stale",
      to: "l48-ctrl",
      label: "fresh 才可消费",
      via: [
        { x: 800, y: 170 },
        { x: 470, y: 170 },
      ],
      labelAt: { x: 650, y: 154 },
    },
    {
      id: "l48-e-row-ctrl",
      from: "l48-row",
      to: "l48-ctrl",
      label: "同一 branch",
      labelAt: { x: 330, y: 286 },
    },
    {
      id: "l48-e-ctrl-out",
      from: "l48-ctrl",
      to: "l48-out",
      label: "取消未来不改历史",
      labelAt: { x: 610, y: 286 },
    },
  ],
  steps: [
    {
      title: "两路输入进入两个时钟",
      description:
        "用户插话走音频块，杯子被挪走走控制周期。同一墙钟上的两件事不得共用一个 available_at 字段。",
      focus: ["l48-in", "l48-audio", "l48-action", "l48-e-in-audio", "l48-e-in-action"],
    },
    {
      title: "各算各的 available_at",
      description:
        "音频列是 block_end + 编码延迟，动作列是 t_obs + 推理延迟 d。评测检查 consumed_at 不早于本列时间戳。",
      focus: ["l48-audio", "l48-action", "l48-stale", "l48-e-audio-stale", "l48-e-action-stale"],
    },
    {
      title: "过期定义不能横着比",
      description:
        "音频过期用一块或一帧的时长，手臂过期用 H/f。同一毫秒延迟可以让一边过期、另一边仍 fresh。",
      focus: ["l48-stale", "l48-e-stale-ctrl"],
    },
    {
      title: "写入同一行，分列保存",
      description:
        "一行状态同时有音频模式和动作模式。PAUSE 只改音频列；力切断只改动作列。两列的 branch 血缘仍可一起查。",
      focus: ["l48-row", "l48-audio", "l48-action", "l48-e-audio-row", "l48-e-action-row"],
    },
    {
      title: "PAUSE 与 REPLAN 分成两次事件",
      description:
        "说话时杯子被挪走：先 PAUSE 停嘴，再 REPLAN 丢弃旧剩余步。一次点击不得同时撤回已播 PCM 和已发生接触。",
      focus: ["l48-ctrl", "l48-row", "l48-e-row-ctrl"],
    },
    {
      title: "只取消未来队列",
      description:
        "REPLAN 取消未播 PCM 和未执行 chunk。已播出的声音和已经发生的接触留在历史上，不能 undo。",
      focus: ["l48-out", "l48-ctrl", "l48-e-ctrl-out"],
    },
  ],
  facts: [
    "音频 available_at = block_end + encode_latency；动作 available_at = t_obs + d。两列不得混比。",
    "音频过期用块长（MiniMind listener 为 320 ms）；动作过期用开环窗口 H/f。",
    "REPLAN 将旧 branch 标为 superseded，未播 PCM 与未执行剩余步都不得再执行。",
    "音频 PAUSE 不等于力切断：PAUSE 不停臂、不进入 SAFE_HOLD，也不撤回接触。",
    "Figure Helix 官方博文写 System 2 为 7–9 Hz、System 1 为 200 Hz。本课不声称复现 Helix 或 GPT-4o。",
  ],
};
