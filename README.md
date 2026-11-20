<!-- DRAFT: not published yet. Repository owner, Hugging Face publisher and consumer-desktop numbers are pending. -->

# moe-tiering

Run big Mixture-of-Experts language models on **one GPU plus system RAM**.

The dense part of the model and the most-used experts live in VRAM. The other experts stay in page-locked RAM, and
the GPU reads them over PCIe with no copy (zero-copy Triton kernels). The experts are compressed with a 2–4.5-bit
trellis code (QTIP/EXL3-style), so a 235B model fits in 24 GB of VRAM plus 64 GB of RAM. An auto-config picks the
compression level and the VRAM/RAM split for the machine it runs on.

**Status: alpha.** The quality results below were each re-run by an independent check (GLM-5.3's not yet). Speed
is the weak point: 1–9 tokens/s on the hardware we measured. Tested on AMD GPUs (ROCm, Linux) and NVIDIA GPUs (CUDA,
Windows).

## Quality: lower error than llama.cpp at the same size

Error is measured as KL divergence from the unquantized (BF16) model's next-token distribution. We use 84 text
windows in four sets: two familiar-text sets, and two sets of new text (recent arXiv abstracts, Vietnamese, Chinese,
Rust/C) that no quantization choice was tuned on. Each row compares our best variant **at or below the size** of a
llama.cpp GGUF of the same model. The percentage is how much lower our KL is, across the four sets.

| Model | ~4 bits vs Q4_K_M | ~3.5 bits vs Q3_K_M | ~2–2.5 bits vs IQ2 |
|---|---|---|---|
| Qwen3-30B-A3B-Thinking-2507 | | | −54 to −60% vs unsloth UD-IQ2_XXS |
| Qwen3-235B-A22B-Instruct-2507 | −15 to −19% vs unsloth Q4_K_M | −58 to −65% vs unsloth UD-Q3_K_XL | −59 to −65% vs bartowski IQ2_M |
| Qwen3.6-35B-A3B | −36 to −42% | −65 to −71% | no size match yet |
| Qwen3.5-122B-A10B | −38 to −47% | −70 to −73% | −69 to −76% vs IQ2_XXS |
| GLM-4.7-Flash | −41 to −53% | −65 to −75% | no size match yet |
| GLM-5.3-Flash | −24 to −33% | −44 to −48% | −31 to −35% vs IQ2_M |
| Gemma-4-26B-A4B | −46 to −53% | −78 to −83% | −74 to −80% vs IQ2_M |
| MiniMax-M2 | −13 to −30% | −36 to −51% | −54 to −60% vs IQ2_XXS |
| gpt-oss-120b, gpt-oss-20b | **we lose**: the release's native MXFP4 experts are better than our re-quantization | | |
| GLM-5.3 (744B) | no GGUF comparison yet (KL vs BF16: 0.029 at 4.5 bits, 0.041 at 3.5, 0.17 at 2.25) | | |
| DeepSeek-V4.1-Flash | runs; quality ladder not measured yet | | |

How the GGUFs were made: Qwen3-30B and Qwen3-235B use the public unsloth / bartowski files. The other models use
GGUFs we made with llama.cpp's own tools and an importance matrix from the same calibration text our converter saw.
Where we are not ahead, the table says so. Two caveats:
- With every non-expert weight kept in BF16, our 4-bit experts alone are level with Q4_K_M's. The 4-bit gains come
  mostly from how we compress the attention and other dense weights.
- KL is a proxy. Downstream benchmark scores (GSM8K, MMLU, ...) are not measured yet.

## What runs where

The auto-config sizes everything from VRAM and RAM. Some measured examples (AMD MI250, PCIe host link):

| Model | VRAM + RAM | Variant picked | Decode speed |
|---|---|---|---|
| Qwen3.6-35B-A3B | 8 GB + 32 GB | 4.5-bit (H4) | 4–6 tok/s |
| Gemma-4-26B-A4B | 8 GB + 32 GB | H4 | 7 tok/s |
| gpt-oss-120b | 16 GB + 32 GB | 2.57-bit | 9 tok/s |
| Qwen3.5-122B-A10B | 8 GB + 32 GB | 2.0-bit | 3.4–5 tok/s |
| MiniMax-M2 (230B) | 24 GB + 64 GB | 2.57-bit | 4.7 tok/s |
| GLM-5.3-Flash | 24 GB + 128 GB | 3.5-bit | 2.25 tok/s |
| GLM-5.3 (744B) | 24 GB + 256 GB | 3.0-bit | 1.1 tok/s |

Consumer-desktop numbers (RTX-class GPU) are not measured yet; they will replace this table.

## Install

Python 3.10+, PyTorch 2.6+ with CUDA or ROCm, and a GPU.

```bash
pip install git+https://github.com/OWNER/moe-tiering          # OWNER: to be decided
# GLM-5.x and DeepSeek-V4.1 need transformers >= 5.19:
pip install "moe-tiering[new-archs] @ git+https://github.com/OWNER/moe-tiering"
```

## Use

```bash
moe-tiering doctor                           # GPU, RAM, library versions
moe-tiering plan  models/Qwen3.6-35B-A3B     # what would run on this machine (nothing is loaded)
moe-tiering run   models/Qwen3.6-35B-A3B     # chat in the terminal
moe-tiering serve models/Qwen3.6-35B-A3B --port 8000
```

`serve` exposes an OpenAI-compatible API (`/v1/chat/completions` with streaming, `/v1/completions`, `/v1/models`):

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 200}'
```

A model directory holds one subdirectory per variant (`H4`, `TQ4.0`, `TQ3.5`, `TQ3.0`, `TQ2.57`, ...) plus the
tokenizer files. `--vram-gb` and `--ram-gb` override the detected budgets, and `--variant` forces one variant.

### Getting models

Pre-converted models will be published on Hugging Face (`moe-tiering pull <model>`); this is not available yet.
Converting a model yourself needs a GPU with enough memory for one layer, a calibration corpus and several hours to
days (`python -m moe_tiering.convert_tcq --help`). The calibration tooling is not packaged yet.

## How it works

- **Experts are tiered.** Each MoE layer keeps a VRAM cache of expert slots. The auto-config either freezes the most
  routed experts in VRAM and serves the rest from pinned RAM, or caches dynamically. Cold experts are read by the
  GPU directly from host memory (zero-copy), so a miss costs PCIe bandwidth, not a copy.
- **Experts are trellis-coded.** Each weight matrix is rotated with a random Hadamard transform, then coded with a
  tail-biting trellis (QTIP/EXL3-style) using full per-expert Hessians from a broad calibration corpus (BlockLDLQ).
  A per-layer rate plan spends bits where the model is most sensitive. The 4.5-bit variant (H4) uses a simpler
  4-bit row code with Hessian-weighted scales.
- **The dense part is compressed separately** (attention, shared experts, embeddings), with architecture-specific
  exceptions where low precision breaks a model.
- **Auto-config** ranks every variant and placement that fits the machine's VRAM and RAM by measured quality and
  picks the best one. RAM headroom for the OS is kept.

## Supported architectures

Qwen3-MoE, Qwen3.5/3.6-MoE, GLM-4.7-Flash, GLM-5.3 and GLM-5.3-Flash, Gemma-4 MoE, MiniMax-M2, gpt-oss, and
DeepSeek-V4.1-Flash (our own port of the reference code; transformers has no implementation).

## Limitations

- Speed: 1–9 tok/s measured, well below what engines that keep the whole model in VRAM reach (no head-to-head
  speed comparison yet). Decode runs one token at a time (batch 1), and big models are limited by PCIe bandwidth to
  the cold experts.
- One request at a time in `serve`; each chat turn re-reads the whole conversation (no prompt cache yet).
- Tested on AMD MI250 (ROCm, Linux) and an NVIDIA GTX 1650 laptop (CUDA, Windows) only.
- 8 GB-VRAM machines cannot run the largest models: some kernels need more memory than that to warm up.

## License

Apache-2.0. Model weights keep their own licenses.
