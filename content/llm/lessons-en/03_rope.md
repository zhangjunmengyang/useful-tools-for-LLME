---
id: rope
title: What RoPE rotates
summary: RoPE does not add a second position vector. It rotates the existing coordinates by position. Farther pairs usually have a weaker dot product.
unit: body
play_tools: [rope_frequencies]
checkpoints:
  - RoPE is a rotation, not an extra concatenated position vector
  - Frequency comes from dimension and base; as distance grows, the relative dot product drops
---

# What RoPE rotates

Self-attention does not know “which word came first”. Shuffle the same content vectors and the scores can stay the same. Position has to be written into the representation.

Early methods added a separate position vector. RoPE (rotary position embedding) rotates pairs of coordinates by an angle that depends on position. Relative position then lives in the difference of two rotations. Far-apart positions spin the high-frequency pairs more, so the dot product usually shrinks. That is one way to get “nearby matters more”. Attention does not invent it on its own.

`base` (often 10000) and the head dimension set how fast each pair turns. Change `base` and the long-range decay curve moves. That is the same family of knobs people retune when a context window suddenly gets longer.

## Learn

Be able to draw one thing: the same coordinate pair at position 0 and position 8, and where the angle differs; then how the relative dot product moves as distance goes from 1 to 8. If you cannot tell “rotate” from “add”, Llama-style configs will not make sense later.

## Play

Defaults are `dim=8` and `max_distance=4`. Run once and look at distance versus dot product in `decay`. Change `base` from 10000 to 500 and run again. Compare whether decay gets steeper or gentler. Ignore whether the absolute values look pretty.
