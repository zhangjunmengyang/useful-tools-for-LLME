---
id: vectors
title: Near and far in vector space
summary: An embedding is a list of coordinates. Nearness is a dot product or a cosine, not a gut feeling that two words “mean the same thing”.
unit: input
play_tools: [vector_similarity]
checkpoints:
  - Similarity compares direction or a dot product, not whether the surface words match
  - Dimensions must match; name the metric before you compare
---

# Near and far in vector space

An embedding writes a piece of text, or one token, as a list of numbers. Attention, retrieval, and classification mostly compare those numbers.

Two common comparisons. The dot product asks how large the projection is. Cosine throws length away and keeps direction. The same vectors can change nearest neighbors if you change the metric. Say which one you used before you say “these two are close”.

Teaching examples often use three vectors: a query, a near one, a far one. Near is not the label `near`. Near is a small angle. Labels can lie. The numbers do not.

This lesson does not train an embedding model. It feeds already-computed coordinates into one function and shows how the matrix ranks them. Real model outputs can go through the same call later.

## Learn

Write three 2-D or 3-D vectors on paper and compute pairwise cosines. Then check the tool’s matrix. If they disagree, check whether you treated a dot product as a cosine.

## Play

The default sample is three 3-D vectors labeled query / near / far. Run it once. Then edit `near` so it matches `far` and see whether the nearest neighbor flips. Change the numbers, then run. Do not only rename the labels.
