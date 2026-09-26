// Expected attention layout of each featured model, read by hand from its config.json.
// hf_id: [full layers, sliding layers, linear layers, window, native format]
export const FEATURED_LAYOUT = {
  'meta-llama/Meta-Llama-3.1-8B-Instruct': [32, 0, 0, null, null],
  'Qwen/Qwen3-8B': [36, 0, 0, null, null],
  'Qwen/Qwen3.5-9B': [8, 0, 24, null, null],
  'google/gemma-3-4b-it': [5, 29, 0, 1024, null],
  'mistralai/Ministral-3-8B-Instruct-2512': [34, 0, 0, null, 'fp8'],
  'google/gemma-4-31B-it': [10, 50, 0, 1024, null],
  'Qwen/Qwen3.6-27B': [16, 0, 48, null, null],
  'Qwen/Qwen3-32B': [64, 0, 0, null, null],
  'mistralai/Mistral-Small-3.2-24B-Instruct-2506': [40, 0, 0, null, null],
  'google/gemma-3-27b-it': [10, 52, 0, 1024, null],
  'microsoft/phi-4': [40, 0, 0, null, null],
  'meta-llama/Llama-3.3-70B-Instruct': [80, 0, 0, null, null],
  'mistralai/Devstral-2-123B-Instruct-2512': [88, 0, 0, null, 'fp8'],
  'openai/gpt-oss-20b': [12, 12, 0, 128, 'mxfp4'],
  'openai/gpt-oss-120b': [18, 18, 0, 128, 'mxfp4'],
  'Qwen/Qwen3.6-35B-A3B': [10, 0, 30, null, null],
  'google/gemma-4-26B-A4B-it': [5, 25, 0, 1024, null],
  'zai-org/GLM-4.5-Air': [46, 0, 0, null, null],
  'Qwen/Qwen3-235B-A22B-Instruct-2507': [94, 0, 0, null, null],
  'deepseek-ai/DeepSeek-V3.2': [61, 0, 0, null, 'fp8'],
};
