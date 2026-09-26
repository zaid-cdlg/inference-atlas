# Changelog

All notable changes to inference-atlas. Versions are `MAJOR.MINOR.PATCH.MICRO`.

## [0.1.0.0] - 2026-09-25

The first public version: open one page and find out whether self-hosting an open-weight LLM
beats paying the API for the same model, and get the command to run it.

### Added

- **Self-host vs API verdict.** Pick a model and a use case (chat, agents or batch) and the page
  tells you at what traffic, in tokens per day, running your own GPUs starts to cost less than
  the API. A chart shows both costs from 100K to 10B tokens per day, with the break-even point
  and your own volume marked. Hover or tap it to read the costs at any volume.
- **The GPU and precision are picked for you.** Auto mode chooses the NVIDIA GPU that breaks
  even soonest, and says when an AMD MI300X might cost less (it needs vLLM's ROCm build). You
  can pick any of 9 GPUs (T4, L4, A10G, L40S, A100, H100, H200, B200, MI300X) and FP16, FP8,
  INT4 or the model's own MXFP4/FP8 weights yourself; options that cannot run say why.
- **A command you can run.** A `vllm serve` line with tensor parallelism, context length, batch
  size and quantization filled in, plus Copy command and Copy link buttons.
- **Why N users fit.** An animated memory bar shows the model's weights and one block per user's
  conversation, and a slider shows how longer conversations fit fewer people.
- **Tune assumptions.** Change the KV cache precision, the longest conversation, input:output
  ratio, cached prefix, how busy the GPUs are, batch size and your tokens per day. Every
  number has a plain-English explanation, and the "How we estimate" section spells out the method.
- **Quality notes.** A cited table of what FP8 and INT4 cost in accuracy, and "(community
  quant)" next to INT4 builds that do not come from the model's publisher.
- **Shareable links.** Every setting lives in the URL, so a copied link reopens the same result.
  Links with unknown or broken values fall back to defaults with a short notice.
- **20 featured open-weight models,** from Llama 3.1 8B to DeepSeek V3.2, with memory maths that
  understands grouped-query, compressed (MLA), sliding-window and linear attention,
  mixture-of-experts, and FP8/MXFP4 checkpoints.
- **A weekly refresh.** Model shapes come from Hugging Face and prices from OpenRouter every
  Monday. A run that looks wrong (too few models, implausible prices, unreadable checkpoints)
  publishes nothing, and models that cannot be sized are left out rather than guessed.
- **Hosting.** The page deploys to GitHub Pages and mirrors to a Hugging Face Space on every
  change to `main`.
- **Accessibility.** Keyboard access with visible focus rings, 44px touch targets, a screen
  reader summary and data table for the chart, reduced-motion support, and a layout that keeps
  the answer above the fold on a phone.

### Notes

- Every number is a first-order estimate and usually optimistic. Measure your own setup with
  `vllm bench serve` before you buy hardware.
