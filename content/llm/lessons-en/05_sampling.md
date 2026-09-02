---
id: sampling
title: How the next token is drawn
summary: Logits are divided by temperature, then cut by top-k / top-p, then sampled. Temperature and truncation are not the same knob.
unit: generate
play_tools: [sampling_distribution]
checkpoints:
  - Temperature changes how peaked the distribution is, not right versus wrong
  - top-k and top-p shrink the candidate set, then you renormalize
---

# How the next token is drawn

The model first emits a logit for every vocabulary item. That is not a probability yet. Typical next steps: divide by temperature, drop some mass with top-k or top-p, then softmax and sample.

Temperature above 1 flattens the distribution, so rare tokens show up more. Temperature near 0 sharpens it, so you almost always draw the max. top-k keeps only the k highest scores. top-p walks from high to low until the cumulative probability reaches p. Both cuts can be combined; the tool’s order is the one this lesson uses.

“Be more creative” that only turns temperature up also turns up nonsense. Cutting the candidate set is often the more direct way to constrain that.

## Learn

Take logits 2, 1, 0. Compute probabilities at temperature 1 and 0.5, then check the tool. After that, set top-k to 2 and see whether the smallest one disappears.

## Play

The default sample is A / B / C. Run temperature 1. Then 0.2 and 1.8. Then set top_p to 0.7. Change one knob at a time and watch the probability table.
