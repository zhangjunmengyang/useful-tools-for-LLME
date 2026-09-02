import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson57Diagram: LessonDiagram = {
  lessonId: "57",
  title: "训练图像的出处准入门",
  summary:
    "候选图像先算 SHA-256，再写入 sidecar 必填字段。许可门与缺字段检查同时失败则拒收。C2PA 软绑定（水印或指纹）可并列记录，不能替代哈希或许可。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l57-in",
      label: ["候选图像"],
      meta: "字节 + URL",
      kind: "input",
      x: 96,
      y: 86,
      width: 132,
    },
    {
      id: "l57-hash",
      label: ["SHA-256"],
      meta: "硬绑定",
      kind: "transform",
      x: 308,
      y: 86,
      width: 132,
    },
    {
      id: "l57-row",
      label: ["sidecar 行"],
      meta: "六项必填",
      kind: "state",
      x: 520,
      y: 86,
      width: 140,
    },
    {
      id: "l57-license",
      label: ["许可门"],
      meta: "空 / unspecified 非法",
      kind: "decision",
      x: 752,
      y: 86,
      width: 156,
    },
    {
      id: "l57-fields",
      label: ["缺字段检查"],
      meta: "URL / 合成 / 撤回",
      kind: "decision",
      x: 308,
      y: 254,
      width: 148,
    },
    {
      id: "l57-soft",
      label: ["C2PA 软绑定"],
      meta: "水印或指纹可选",
      kind: "transform",
      x: 520,
      y: 254,
      width: 148,
    },
    {
      id: "l57-out",
      label: ["训练集或拒收"],
      meta: "缺许可或缺哈希拒",
      kind: "output",
      x: 752,
      y: 254,
      width: 156,
    },
  ],
  edges: [
    {
      id: "l57-e-in-hash",
      from: "l57-in",
      to: "l57-hash",
      label: "读像素字节",
      labelAt: { x: 198, y: 54 },
    },
    {
      id: "l57-e-hash-row",
      from: "l57-hash",
      to: "l57-row",
      label: "写入 sha256",
      labelAt: { x: 410, y: 54 },
    },
    {
      id: "l57-e-row-license",
      from: "l57-row",
      to: "l57-license",
      label: "查允许集合",
      labelAt: { x: 640, y: 54 },
    },
    {
      id: "l57-e-row-fields",
      from: "l57-row",
      to: "l57-fields",
      label: "查空字段",
      via: [
        { x: 520, y: 170 },
        { x: 308, y: 170 },
      ],
      labelAt: { x: 390, y: 154 },
    },
    {
      id: "l57-e-hash-soft",
      from: "l57-hash",
      to: "l57-soft",
      label: "可选，不替代",
      via: [
        { x: 308, y: 170 },
        { x: 520, y: 170 },
      ],
      labelAt: { x: 430, y: 186 },
    },
    {
      id: "l57-e-license-out",
      from: "l57-license",
      to: "l57-out",
      label: "许可通过才继续",
    },
    {
      id: "l57-e-fields-out",
      from: "l57-fields",
      to: "l57-out",
      label: "字段齐全",
      via: [
        { x: 308, y: 320 },
        { x: 752, y: 320 },
      ],
      labelAt: { x: 520, y: 336 },
    },
    {
      id: "l57-e-soft-out",
      from: "l57-soft",
      to: "l57-out",
      label: "记录来源类型",
      labelAt: { x: 640, y: 286 },
    },
  ],
  steps: [
    {
      title: "先对图像字节做硬绑定",
      description:
        "SHA-256 把任意长度字节映到 256 bit。C2PA 2.4 允许新建哈希用 sha256、sha384、sha512。空哈希非法。",
      focus: ["l57-in", "l57-hash", "l57-e-in-hash"],
    },
    {
      title: "写成一行 sidecar，而不是只放文件路径",
      description:
        "必填字段是 sample_id、source_url、license、sha256、is_synthetic、retractable。路径出现在列表里不等于字段齐全。",
      focus: ["l57-row", "l57-hash", "l57-e-hash-row"],
    },
    {
      title: "许可门拒绝空值和 unspecified",
      description:
        "Data Provenance Initiative 在托管站上看到 70%+ 许可未标明。本课把空许可、unspecified、Unknown 都判非法。",
      focus: ["l57-license", "l57-row", "l57-e-row-license"],
    },
    {
      title: "缺字段的行不得进训练集",
      description:
        "来源 URL、合成标记、可撤回标记缺一项即拒收。合成标记为真也不能补上许可或哈希。",
      focus: ["l57-fields", "l57-e-row-fields", "l57-e-fields-out", "l57-out"],
    },
    {
      title: "软绑定可记，不能替代准入谓词",
      description:
        "C2PA 把水印和指纹叫做软绑定。规格不把检测率写成硬条件。C2PA 2.0 已删除 Training and Data Mining assertion。",
      focus: ["l57-soft", "l57-e-hash-soft", "l57-e-soft-out", "l57-license"],
    },
  ],
  facts: [
    "C2PA 2.4 第 13.1 节允许的新建哈希算法是 SHA2-256（sha256）、SHA2-384（sha384）、SHA2-512（sha512）；SHA-3 不在允许列表。",
    "C2PA 规格写明：它不判断 provenance 数据好坏，只判断断言能否与资产绑定、格式正确、未被篡改。",
    "Data Provenance Initiative（arXiv:2310.16787）审计 1858 个文本数据集：GitHub / Hugging Face / Papers with Code 上未标明许可分别为 72% / 69% / 70%。",
    "C2PA 2.0（2024-01）删除了 Training and Data Mining assertion。训练选择权不在当前 Content Credential 里。",
    "LAION-5B 论文写明发布的是 5.85B 条 CLIP 过滤后的图文 URL 对，元数据 CC-BY-4.0，不拥有图像版权。",
  ],
};
