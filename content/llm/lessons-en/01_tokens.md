---
id: tokens
title: How text becomes tokens
summary: The model does not read characters. It reads integer IDs. Change the tokenizer and both the IDs and the length change.
unit: input
play_tools: [unicode_analyze, tokenizer_encode]
checkpoints:
  - Full-width and half-width letters are different Unicode code points
  - The same sentence can have very different token counts under two tokenizers
---

# How text becomes tokens

A model does not see “words”. It sees a list of integers. Those integers come from three steps: split the string into Unicode code points, merge them into subwords with a tokenizer, then look up IDs in a vocabulary.

Two layers get mixed up. Unicode answers “which code is this character”. The tokenizer answers “how are those codes cut into pieces the model knows”. Full-width `Ａ` and half-width `A` look like the same letter; they are not the same code point. Some tokenizers run NFKC first. Some do not. So the same visible text can enter the model at different lengths.

Tokenizers usually merge frequent pieces with BPE or a close cousin. Common English words often stay in one piece. Rare words and Chinese often shatter. That is not a bug. It is what the vocabulary saw in training.

## Learn

Keep the two layers apart:

1. Character layer: paste text with full-width letters, accents, and spaces. Look at each code point’s category and byte length.
2. Token layer: take the same mixed Chinese–English sentence, switch vocabularies, and count tokens.

Ask: did the length change because the code points changed, or because the cuts changed? You are done only after you have seen both.

## Play

Both tools are real API calls, not cartoons.

1. Drop `Ａ café` into Unicode analysis. Check the code points for full-width A and `é`, and whether NFC still equals NFKC.
2. If a vocabulary is cached or you can download one, encode a mixed sentence with `openai-community/gpt2`. Write down the token count. Then try a paraphrase and see how length moves.

`tokenizer_encode` pulls a Hugging Face vocabulary. The first run can be slow. If you only want the character layer, the first tool is enough.
