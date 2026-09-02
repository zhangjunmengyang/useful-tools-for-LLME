---
id: activations
title: 前馈层在用哪条激活曲线
summary: GELU、SiLU、SwiGLU 不是换皮。负半轴怎么漏、门控怎么乘，决定前馈层记住什么。
unit: body
play_tools: [ffn_activation_compare]
checkpoints:
  - ReLU 砍掉负值；GELU 和 SiLU 在负半轴漏一点
  - SwiGLU 是门控，不是单条曲线换名字
---

# 前馈层在用哪条激活曲线

Transformer 每个块里，注意力后面还有一层前馈网络（FFN）。它按位置独立地做一次非线性变换。用哪条激活函数，决定负值还留不留、要不要再乘一门。

ReLU 简单：小于 0 就置零。GELU 和 SiLU 在 0 附近更软，负半轴会漏过一小截。SwiGLU 不是再画一条 S 形，而是两条通路相乘：一条当门，一条当内容。Llama 一类模型用的就是这种。

课里不推导谁在哪个任务上一定更好。只要求你能看图说出：同一段输入，四条曲线在负半轴和零点附近差在哪。后面读模型配置里的 `hidden_act`，才知道改的是哪条线。

## 学

默画四条线的零点附近。能指出哪条完全砍负值、哪条是两条通路相乘，就算过。

## 玩

用默认的 `x_values` 跑一遍，看返回的 `activations`。把输入改成更密的一串负数到正数，再看 SwiGLU 和其他三条是不是同一类形状。
