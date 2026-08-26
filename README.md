# ComfyUI-Breeze-TTS-2

[Breeze TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) nodes for ComfyUI: **voice clone**,
**voice design**, and **voice direction** with bilingual (EN/ZH) speech, inline vocal events,
INT8 ConvRot quantized builds, and full ComfyUI / AIMDO dynamic-VRAM integration.

## Nodes (category: `Breeze TTS 2`)

| Node | Purpose |
| --- | --- |
| **Breeze TTS 2 Load Model** | Loads the checkpoint with `dtype` (auto/bf16/fp32), `device` (auto/cuda/cpu), `attention` (auto/sdpa/flash_attention/sageattention), and `download_if_missing`. |
| **Breeze TTS 2 Voice Clone** | Clones a speaker from clean reference audio + its exact transcript (CFG 1.0). |
| **Breeze TTS 2 Voice Design** | Creates a voice from a natural-language description, no reference audio (CFG 4). |
| **Breeze TTS 2 Voice Direction** | Clones a reference voice and steers tone, emotion, pace, and delivery with an instruction (CFG 4). |
| **Breeze TTS 2 Whisper Transcribe** | Whisper helper that produces the exact transcript cloning needs. |

Vocal events work inline in the text: `(laugh)`, `(cough)`, `(clears throat)`, `(sigh)` in English;
`[笑]`, `[咳嗽]`, `[清嗓子]`, `[叹气]` in Chinese. Match the instruction language to the text language
for design/direction. Output is 24 kHz mono.

## Checkpoints

The loader downloads into `ComfyUI/models/breezetts2/` when `download_if_missing` is enabled:

| Choice | File | Notes |
| --- | --- | --- |
| int8 hybrid **(default)** | `Breeze-TTS-2-int8-hybrid.safetensors` | Backbone + text encoder INT8 ConvRot, depth decoder bf16. bf16-level speed at −27 % VRAM. |
| bf16 combined | `Breeze-TTS-2-bf16.safetensors` | Same weights merged into one file (bit-exact). Fastest. |
| INT8 all linears | `Breeze-TTS-2-int8-convrot.safetensors` | Smallest build; slower in the decode loop (see below). |
| INT8 text encoder only | `Breeze-TTS-2-int8-text-encoder.safetensors` | Decode path fully bf16; smallest quality-irrelevant VRAM cut. |

All mirrors live in [drbaph/Breeze-TTS-2-comfyui](https://huggingface.co/drbaph/Breeze-TTS-2-comfyui).

## Model Files

Weights are stored per source repo under `ComfyUI/models/breezetts2/` (primary folder first; extra_model_paths.yaml entries and symlinks are searched too). Missing files download automatically when `download_if_missing` is on, otherwise the error names the expected folder.

```
📂 ComfyUI/
└── 📂 models/
    └── 📂 breezetts2/
        └── 📂 drbaph_Breeze-TTS-2-comfyui/      (all builds, default source)
            ├── Breeze-TTS-2-bf16.safetensors              (6.49 GiB, official shards merged bit-exact)
            ├── Breeze-TTS-2-int8-hybrid.safetensors       (4.53 GiB, recommended int8 build)
            ├── Breeze-TTS-2-int8-convrot.safetensors      (4.22 GiB, all 462 linears quantized)
            ├── Breeze-TTS-2-int8-text-encoder.safetensors (5.84 GiB, decode path bf16)
            ├── config.json
            ├── generation_config.json
            ├── tokenizer.json
            ├── tokenizer_config.json
            ├── special_tokens_map.json
            └── 📂 audio_tokenizer/                        (Qwen3-TTS 12 Hz codec, shared)
                ├── model.safetensors                      (0.64 GiB)
                ├── config.json
                ├── configuration.json
                └── preprocessor_config.json
```

Only the selected weights file is downloaded; every build shares the same `config.json`, tokenizer files, and `audio_tokenizer/` codec.

| Component | bf16 build | int8 builds |
| --- | --- | --- |
| Model weights | 6.49 GiB | 4.22–5.84 GiB |
| `audio_tokenizer` codec | 0.64 GiB | unchanged | unchanged |
| Tokenizer + configs | ~35 MB | unchanged | unchanged |

Expect roughly **5.3–7.5 GiB VRAM** depending on build and dtype; AIMDO DynamicVRAM pages castable weights to keep live VRAM pressure low alongside other models.

## Benchmarks (RTX 5090, ComfyUI loader, sdpa)

| Build | avg RTF | Peak VRAM | Whisper WER vs prompt |
| --- | --- | --- | --- |
| bf16 | 5.68 | 7.51 GiB | baseline |
| int8-convrot (all linears) | 9.08 | 5.21 GiB | matches bf16 |
| **int8-hybrid (depth decoder bf16)** | **5.64** | **5.53 GiB** | matches bf16 |
| int8-text-encoder only | 5.34 | 6.82 GiB | matches bf16 |

Why full INT8 regresses: the depth decoder runs 15 skinny single-token GEMMs per 12.5 Hz frame
(≈1260 linear calls/frame). At M=1–2 the int8 per-call overhead (~0.1 ms quantize + dispatch +
allocations) dwarfs the GEMM savings, so those layers lose to plain bf16 `F.linear` (~0.03 ms).
The backbone contributes only ~196 calls/frame and the text encoder runs once at prefill, so
quantizing them is nearly free — that is exactly what the hybrid build does. Full measurement
data is in `benchmark_report.json` on the mirror repo.

## Requirements

- ComfyUI with **Transformers 4.57 or 5.3+**. On 5.x the text encoder uses the native
  `T5Gemma2TextEncoder` (which has no flash-attention kernel — it falls back to sdpa when
  `flash_attention` is selected); on 4.x it uses the vendored upstream compat implementation,
  which does run flash attention.
- torch/torchaudio from your ComfyUI install. `flash_attn` and `sageattention` are optional.
- `comfy-kitchen` ships with current ComfyUI and is only imported when an INT8 checkpoint is loaded.

## Model architecture (what actually runs)

- **Text encoder**: T5Gemma2 (26 layers, 1152 hidden) + linear projection into the backbone space.
- **Backbone**: Qwen3-style decoder (28 layers, 2048 hidden) predicting the first codebook token per
  12.5 Hz frame, with an audio-token embedding summed over the 16 codebooks.
- **Depth decoder**: 12-layer decoder generating codebooks 1–15 per frame with its own CFG.
- **Audio tokenizer**: vendored Qwen3-TTS 12 Hz codec (Mimi-style encoder + streaming decoder),
  24 kHz, 16 codebooks. Reference audio is encoded with the same codec.
- The in-checkpoint `codec_model` (a Mimi copy used only by the upstream training forward) is not
  built at inference.

The eager generation loop reimplements the official `FastBreezeStreamingRuntime` with all fast
stages disabled (DynamicCache instead of CUDA graphs) so it stays compatible with AIMDO dynamic
VRAM paging and INT8 weight casting. The official CUDA-graph fast path is intentionally not ported.

## License

Node-pack code: Apache-2.0 (see `LICENSE`). Vendored model code keeps its upstream licenses
(Apache-2.0 from breezeblue-ai/breeze-tts and Qwen/Qwen3-TTS).

**Model weights, derivative checkpoints, and self-hosted outputs are governed by the BreezeBlue
Research and Non-Commercial License** — see [MODEL_LICENSE](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE).
Commercial use requires written authorization from RESONIA, INC.
