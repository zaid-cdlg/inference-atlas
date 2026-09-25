# GPU spec and price sources

All values checked 2026-09-25.

## Method

- **Specs:** official NVIDIA and AMD product pages or datasheets only.
- **FLOPS are DENSE.** No structured sparsity. Where the vendor shows only a sparse figure
  (H100, H200, L4, B200), dense = sparse / 2, which is what the vendor's own footnote says.
  T4 (Turing) has no sparsity feature, so its figure is dense as printed.
- **Prices:** on-demand, USD per GPU per hour, cheapest single-GPU instance per provider where one
  exists. RunPod quotes use **Secure Cloud** (data-centre tier), not Community Cloud. The value
  in `gpus.json` is the median of the quotes below (mean of the middle two when the count is even).
- **Primary** = read from the provider's own pricing page. **Secondary** = read from the
  getdeploying.com aggregator; those rows carry `verify: true` in the JSON.
- Instances bundle different CPU/RAM, so prices are indicative only.

## Specs

| GPU | VRAM | Bandwidth | FP16 dense TFLOPS | FP8 dense TFLOPS | Printed on source | Source |
|---|---|---|---|---|---|---|
| T4 | 16 GB | 320 GB/s ("320+") | 65 | n/a | "65 FP16 TFLOPS" mixed precision | [nvidia.com/tesla-t4](https://www.nvidia.com/en-us/data-center/tesla-t4/) |
| L4 | 24 GB | 300 GB/s | 121 | 242.5 | 242 / 485 "shown with sparsity, one-half lower without" | [nvidia.com/l4](https://www.nvidia.com/en-us/data-center/l4/) |
| A10G | 24 GB | 600 GB/s | 70 | n/a | Baseten: "A10G only has 70 TF" (A10 datasheet: 125 dense) | [aws g5](https://aws.amazon.com/ec2/instance-types/g5/) (VRAM), [nvidia.com/a10](https://www.nvidia.com/en-us/data-center/products/a10-gpu/) (bandwidth), [Baseten A10 vs A10G](https://www.baseten.co/blog/nvidia-a10-vs-a10g-for-ml-model-inference/) (FLOPS) |
| L40S | 48 GB | 864 GB/s | 362.05 | 733 | "362.05 \| 733*", "733 \| 1,466*" | [nvidia.com/l40s](https://www.nvidia.com/en-us/data-center/l40s/) |
| A100 80GB SXM | 80 GB | 2,039 GB/s | 312 | n/a | "312 \| 624*" | [nvidia.com/a100](https://www.nvidia.com/en-us/data-center/a100/) |
| H100 SXM | 80 GB | 3,350 GB/s | 989.5 | 1,979 | 1,979 / 3,958 "with sparsity" | [nvidia.com/h100](https://www.nvidia.com/en-us/data-center/h100/) |
| H200 SXM | 141 GB | 4,800 GB/s | 989.5 | 1,979 | 1,979 / 3,958 "with sparsity" | [nvidia.com/h200](https://www.nvidia.com/en-us/data-center/h200/) |
| B200 | 180 GB | 8,000 GB/s | 2,250 | 4,500 | DGX B200: 1,440 GB, 64 TB/s (8 GPUs). HGX B200: FP16 36 PF, FP8 72 PF sparse (8 GPUs), "dense is 1/2" | [nvidia.com/dgx-b200](https://www.nvidia.com/en-us/data-center/dgx-b200/), [nvidia.com/hgx](https://www.nvidia.com/en-us/data-center/hgx/) |
| MI300X | 192 GB | 5,300 GB/s | 1,307.4 | 2,614.9 | dense column of AMD table (sparse 2,614.9 / 5,229.8) | [AMD MI300X datasheet PDF](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf) |

**Flags**

- **A10G (verify, only non-vendor spec):** NVIDIA publishes no A10G datasheet; it is an
  AWS-only variant of the A10. VRAM (24 GB) is from AWS. Bandwidth 600 GB/s is the A10 value
  (both share memory specs). FP16 tensor uses Baseten's 70 TFLOPS instead of the A10's 125,
  because using 125 would overstate prefill speed. AWS only states "up to 250 TOPS" per GPU.
- **B200 (checked):** per-GPU values are 8-GPU system totals divided by 8. Early B200 material
  said 192 GB per GPU; current DGX/HGX pages say 1,440 GB per 8 GPUs (180 GB). RunPod also
  lists B200 at 180 GB. Used 180.
- **A100:** SXM figures used (PCIe 80GB is 1,935 GB/s). Prices mix SXM and "A100 80GB" listings.
- **H100/H200:** SXM figures. PCIe and NVL variants differ.

## Prices (USD per GPU per hour, on-demand)

### T4
| Provider | Instance | $/GPU/hr | Type |
|---|---|---|---|
| Hugging Face | Spaces t4-small | 0.40 | primary ([hf.co/pricing](https://huggingface.co/pricing)) |
| Google Cloud | n1-standard-4 + T4 | 0.42 | secondary ([getdeploying](https://getdeploying.com/gpus/nvidia-t4)) |
| AWS | g4dn.xlarge us-east-1 Linux | 0.526 | AWS list price via [Vantage](https://instances.vantage.sh/aws/ec2/g4dn.xlarge) (AWS page renders prices in JS) |
**Median: 0.42** (verify: the GCP quote is secondary; Google's price page would not load)

### L4
| Provider | Instance | $/GPU/hr | Type |
|---|---|---|---|
| RunPod | L4 Secure Cloud | 0.49 | primary ([runpod.io/pricing](https://www.runpod.io/pricing)) |
| Hugging Face | L4 1x | 0.80 | primary ([hf.co/pricing](https://huggingface.co/pricing)) |
**Median: 0.65** (0.645 rounded)

### A10G
| Provider | Instance | $/GPU/hr | Type |
|---|---|---|---|
| Hugging Face | Spaces a10g-small | 1.00 | primary ([hf.co/pricing](https://huggingface.co/pricing)) |
| AWS | g5.xlarge us-east-1 Linux | 1.006 | AWS list price via [Vantage](https://instances.vantage.sh/aws/ec2/g5.xlarge) |
**Median: 1.00** (1.005 rounded down). Only AWS and HF sell A10G.

### L40S
| Provider | Instance | $/GPU/hr | Type |
|---|---|---|---|
| RunPod | L40S Secure Cloud | 1.09 | primary |
| DigitalOcean | L40S single-GPU droplet | 1.57 | primary ([digitalocean.com](https://www.digitalocean.com/pricing/gpu-droplets)) |
| Hugging Face | L40S 1x | 1.80 | primary |
**Median: 1.57**. RunPod Community is 0.79.

### A100-80GB
| Provider | Instance | $/GPU/hr | Type |
|---|---|---|---|
| RunPod | A100 SXM 80GB Secure Cloud | 1.59 | primary |
| Hugging Face | A100 large (1x, 80GB) | 2.50 | primary |
| Lambda | 8x A100 SXM 80GB (per GPU) | 2.79 | primary ([lambda.ai/pricing](https://lambda.ai/pricing)) |
**Median: 2.50**. Lambda sells A100 80GB only as 8x.

### H100
| Provider | Instance | $/GPU/hr | Type |
|---|---|---|---|
| RunPod | H100 SXM Secure Cloud | 3.49 | primary |
| Lambda | 1x H100 SXM | 4.29 | primary (8x is 3.99/GPU) |
| DigitalOcean | HGX H100 single-GPU droplet | 4.41 | primary ([digitalocean.com](https://www.digitalocean.com/pricing/gpu-droplets)) |
| Hugging Face | Inference Endpoints H100 1x | 4.50 | primary |
**Median: 4.35** (mean of middle two, 4.29 and 4.41)

### H200
| Provider | Instance | $/GPU/hr | Type |
|---|---|---|---|
| DigitalOcean | HGX H200 single-GPU droplet | 4.47 | primary ([digitalocean.com](https://www.digitalocean.com/pricing/gpu-droplets)) |
| RunPod | H200 Secure Cloud | 4.59 | primary |
| Hugging Face | Inference Endpoints H200 1x | 5.00 | primary |
**Median: 4.59**. Aggregator median across 36 providers is also 4.50. Lambda lists no H200.

### B200
| Provider | Instance | $/GPU/hr | Type |
|---|---|---|---|
| RunPod | B200 Secure Cloud | 6.79 | primary |
| Lambda | 1x B200 SXM6 | 6.99 | primary (8x is 6.69/GPU) |
| Hugging Face | Inference Endpoints B200 1x | 9.25 | primary |
**Median: 6.99**

### MI300X
| Provider | Instance | $/GPU/hr | Type |
|---|---|---|---|
| RunPod | 1x MI300X Secure Cloud | 2.39 | primary ([runpod.io/gpu-models/mi300x](https://www.runpod.io/gpu-models/mi300x)) |
| DigitalOcean | MI300X single-GPU droplet | 2.59 | primary ([digitalocean.com](https://www.digitalocean.com/pricing/gpu-droplets)) |
| Hot Aisle | 1x MI300X VM, new customers | 2.99 | primary ([hotaisle.xyz/pricing](https://www.hotaisle.xyz/pricing/)) |
**Median: 2.59**. Cross-checked against [getdeploying](https://getdeploying.com/gpus/amd-mi300x).
Hyperscalers are far higher (Azure 7.86, Oracle 6.00 per GPU, 8x only); excluded as outliers
for a beginner single-GPU choice.

## Not found / skipped

- Google Cloud official GPU price page did not load fully, so GCP numbers are secondary.
- No official A10G spec sheet exists.
