---
id: lora
title: What LoRA actually changes
summary: LoRA adds a low-rank pair on chosen projections. It saves trainable parameters. It does not let you forget the original weights at inference.
unit: adapt
play_tools: [lora_params_estimate, training_cost_estimate]
checkpoints:
  - Rank and target modules set how many new parameters you add
  - Training cost is a second bill: tokens, GPUs, MFU. Fewer LoRA parameters are not free compute
---

# What LoRA actually changes

Full fine-tuning writes the whole weight file. LoRA writes a thinner update: on chosen modules (often the attention Q and V projections) it adds skinny matrices `A` and `B`, and uses their product as the update. The original weights can stay frozen.

Trainable parameters are on the order of layers × target modules × `2 × hidden × rank`. Rank 8 versus 64 is capacity, memory, and overfitting risk. Training only Q and V versus also the MLP is another jump.

LoRA is not “training for free”. The forward pass still runs the full model. If the token count stays the same, the compute bill stays. What you save is mostly optimizer state and writable parameters. Training cost needs its own numbers: parameter count, tokens, GPU throughput, MFU, and the hourly price.

## Learn

Estimate one case: 32 layers, hidden size 4096, only `q_proj` / `v_proj`, rank 16. Get the order of magnitude right. You do not need the last digit.

## Play

Run the LoRA estimator and write down the new parameter count. Change `rank` to 64, then add `k_proj` to `target_modules`. Watch the jump. Then use the training-cost tool and change only `tokens` or `mfu`. Those two tools answer different questions. Do not add them into one “total bill”.
