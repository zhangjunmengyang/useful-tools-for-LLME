---
id: activations
title: Which activation the FFN uses
summary: GELU, SiLU, and SwiGLU are not skins on the same curve. How the negative half leaks, and whether a gate multiplies, decides what the feed-forward layer keeps.
unit: body
play_tools: [ffn_activation_compare]
checkpoints:
  - ReLU zeros the negative side; GELU and SiLU leak a little
  - SwiGLU is a gate, not one renamed curve
---

# Which activation the FFN uses

After attention, each Transformer block has a feed-forward network. It applies a nonlinearity independently at each position. The activation decides whether negatives survive and whether a second path multiplies as a gate.

ReLU is blunt: below zero becomes zero. GELU and SiLU are softer near zero and leak a sliver on the negative side. SwiGLU is not another S-curve. It multiplies two paths: one gate, one content. Llama-style models use that.

This lesson does not crown a winner on a benchmark. You should be able to look at a plot and say how the four curves differ on the negative side and near zero. Then `hidden_act` in a config is a line you can name, not a string you skip.

## Learn

Sketch the four curves near zero. Point to the one that fully cuts negatives, and the one that multiplies two paths.

## Play

Run the default `x_values` and inspect `activations`. Then feed a denser sweep from negative to positive and check whether SwiGLU still looks like the other three.
