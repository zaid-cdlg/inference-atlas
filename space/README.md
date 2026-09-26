---
title: inference-atlas
emoji: 🧮
colorFrom: red
colorTo: gray
sdk: static
pinned: false
license: mit
short_description: Self-host an open LLM or pay the API? Find the break-even
---

# inference-atlas

Should you self-host this LLM, or just pay the API? Pick an open-weight model and a use case to
see which GPUs fit it, how many users fit at once, the cost per million tokens, where
self-hosting beats the API, and the `vllm serve` command to run it.

This Space mirrors the page at https://zaid-cdlg.github.io/inference-atlas/ and is updated by
the weekly refresh. Source, method and data sources: https://github.com/zaid-cdlg/inference-atlas

Estimates are first-order and usually optimistic. Verify with `vllm bench serve` before you buy
anything.
