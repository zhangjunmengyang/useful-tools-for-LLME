---
id: kv-cache
title: Why KV cache grows as you generate
summary: Each new token leaves its key and value in every layer. Length grows, cache grows. The weight file did not get larger.
unit: generate
play_tools: [kv_cache_growth, kv_cache_estimate]
checkpoints:
  - Cache scales with layers, heads, hidden size, sequence length, and precision
  - Generating 16 steps versus 256 steps changes the cache, not a full recompute of attention
---

# Why KV cache grows as you generate

While decoding, every step of attention must see the prefix already written. Recomputing that prefix from scratch each time would grow time and memory with the square of length. The usual fix is to keep keys and values already computed. That table is the KV cache.

Cache size scales with layers, heads, head dimension, current sequence length, batch, and numeric precision. A long prompt fills a chunk before generation starts. Each new token appends another slice.

So “the same 7B model, context from 2K to 32K” often blows up on cache first, not on whether the weights fit. Quantizing weights and quantizing cache are also not the same job.

## Learn

Say it in your own words: weights are one copy; the cache is a separate table sized to this conversation. If you cannot split those two bills, later memory estimates will not add up.

## Play

Defaults assume 32 layers, hidden size 4096, a 1024-token prompt, then 16 generated steps. Run `kv_cache_growth` and watch `cache_gb` per step. Then set `generation_length` to 128. Compare the increment, not whether the absolute number matches a vendor slide.
