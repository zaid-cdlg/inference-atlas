# inference-atlas

<img src="assets/hero.webp" width="1600" height="900" alt="Pencil drawing of a graphics card shaped like a water dispenser. Its clear tank holds a grey block for the model's weights and small coral cubes stacked above it, one cube per user, with empty space left at the top.">

For popular models the API is often cheaper. inference-atlas shows you where that flips.

**Should you self-host this LLM, or just pay the API?**

inference-atlas is a free calculator you open in your browser. Nothing to install. Pick a model
and a use case, and it shows you:

- **Self-host vs API break-even:** the tokens per day at which running your own GPUs becomes
  cheaper than paying for the same model through an API.
- **Model, GPU and precision:** which GPUs fit the model, how many you need, and what FP16, FP8
  or INT4 trades away.
- **Users and throughput:** how many people can use it at once, how fast it answers, and the cost
  per million tokens.
- **A ready-to-run command:** a `vllm serve` command with the settings filled in.

The model list refreshes every week, so new open-weight models show up on their own.

<picture>
  <source media="(prefers-reduced-motion: reduce)" srcset="assets/kv-sim-poster.png">
  <img src="assets/kv-sim.gif" width="800" height="275" alt="The page's memory bar for one GPU: a grey block for the model's weights, then coral blocks filling in, one per user, until 68 users fit. The conversation-length slider is dragged from 4k to 16k tokens and the bar re-fills with fewer, longer blocks until 17 users fit.">
</picture>

*Llama 3.3 70B on one NVIDIA B200 (auto-picked, FP8), chat use case with the longest conversation raised to 16k tokens: 68 users fit at 4k tokens each, 17 at 16k. [Open this setup](https://zaid-cdlg.github.io/inference-atlas/?model=meta-llama%2FLlama-3.3-70B-Instruct&use=chat&max_ctx=16384).*

<!-- BUTTON: "Open the calculator" link goes here -->

> Estimates are first-order and usually optimistic. Verify on your own hardware with
> `vllm bench serve` before you buy anything.

## Related tools

LLMScale and SelfHostLLM check whether a model fits in VRAM. NVIDIA AIConfigurator and BentoML
llm-optimizer tune serving configs for experts. inference-atlas adds the self-host vs API
break-even and a runnable `vllm serve` command, for beginners.

## License

MIT. By Zaid Ibrahim.
