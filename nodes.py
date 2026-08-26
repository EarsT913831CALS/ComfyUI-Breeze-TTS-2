"""ComfyUI nodes for Breeze TTS 2."""

from __future__ import annotations

import logging
from typing import Any

import torch

from . import loader
from . import native
from . import runtime
from .loader import ATTENTION_OPTIONS, DEVICE_OPTIONS, DTYPE_OPTIONS, REPO_CHOICES

logger = logging.getLogger("BreezeTTS2")

CATEGORY = "Breeze TTS 2"
PROGRESS_UNITS_PER_GENERATION = 1000

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    from comfy.utils import ProgressBar
except Exception:
    ProgressBar = None


def get_model_choices() -> list[str]:
    return list(REPO_CHOICES.keys())


def _text_input(default: str, tooltip: str) -> tuple:
    return ("STRING", {"default": default, "multiline": True, "tooltip": tooltip})


def _generation_controls() -> dict:
    return {
        "max_new_tokens": (
            "INT",
            {
                "default": 1500,
                "min": 64,
                "max": 3000,
                "step": 8,
                "tooltip": "Maximum audio frames to generate (12.5 frames per second of speech; the model stops at EOS by itself).",
            },
        ),
        "temperature": (
            "FLOAT",
            {"default": 0.9, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Backbone sampling temperature."},
        ),
        "top_k": ("INT", {"default": 50, "min": 0, "max": 1024, "tooltip": "Backbone top-k (0 disables)."}),
        "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Backbone top-p (1.0 disables)."}),
        "repetition_penalty": (
            "FLOAT",
            {"default": 1.1, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "HF-style repetition penalty on generated backbone tokens."},
        ),
        "depth_temperature": (
            "FLOAT",
            {"default": 0.9, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Depth decoder (codebook 1-15) sampling temperature."},
        ),
        "depth_top_k": ("INT", {"default": 50, "min": 0, "max": 1024, "tooltip": "Depth decoder top-k (0 disables)."}),
        "depth_top_p": (
            "FLOAT",
            {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Depth decoder top-p (1.0 disables)."},
        ),
        "seed": (
            "INT",
            {"default": 42, "min": 0, "max": 2**31 - 1, "tooltip": "0 uses the current random state. A positive value is repeatable."},
        ),
    }


def _cfg_input(default: float, tooltip: str) -> tuple:
    return (
        "FLOAT",
        {"default": default, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": tooltip},
    )


class BreezeTTS2LoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    get_model_choices(),
                    {
                        "default": loader.HYBRID_LABEL,
                        "tooltip": (
                            "int8 hybrid: backbone + text encoder INT8, depth decoder bf16 — bf16 speed at 5.5 GiB (recommended).\n"
                            "bf16: no quantization — best quality, 7.5 GiB.\n"
                            "int8: all transformer linears INT8 — smallest at 5.2 GiB, ~60% slower decode.\n"
                            "int8 text encoder only: decode path stays bf16 — 6.8 GiB."
                        ),
                    },
                ),
                "dtype": (
                    DTYPE_OPTIONS,
                    {"default": "auto", "tooltip": "auto picks bf16 on supporting GPUs, fp32 otherwise."},
                ),
                "device": (DEVICE_OPTIONS, {"default": "auto", "tooltip": "auto uses ComfyUI's active torch device."}),
                "attention": (
                    ATTENTION_OPTIONS,
                    {"default": "auto", "tooltip": "auto uses flash_attention_2 when flash_attn is installed, else sdpa."},
                ),
                "download_if_missing": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Download the selected checkpoint from Hugging Face when missing locally."},
                ),
            }
        }

    RETURN_TYPES = ("BREEZE_TTS2_MODEL",)
    RETURN_NAMES = ("breeze_model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Load a Breeze TTS 2 checkpoint with dtype, device, and attention selection."

    def load(self, model, dtype, device, attention, download_if_missing):
        bundle = loader.load_breeze_bundle(model, dtype, device, attention, bool(download_if_missing))
        return (bundle,)


def _generate_audio(
    bundle,
    *,
    text: str,
    instruction: str,
    ref_audio: dict | None,
    ref_text: str | None,
    cfg_scale: float,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    depth_temperature: float,
    depth_top_k: int,
    depth_top_p: float,
    seed: int,
) -> dict:
    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")
    loader.resume_bundle_to_device(bundle)
    device = bundle.device

    ref_codes = None
    if ref_audio is not None:
        if not ref_text or not ref_text.strip():
            raise ValueError("Reference audio requires its exact transcript (ref_text).")
        wav, sample_rate = runtime.comfy_audio_to_tensor(ref_audio)
        if wav.numel() == 0:
            raise ValueError("Reference audio is empty.")
        ref_codes = runtime.encode_reference_audio(bundle.codec, wav, sample_rate)
        ref_seconds = ref_codes.shape[0] / runtime.FRAMES_PER_SECOND
        if ref_seconds > runtime.MAX_REFERENCE_SECONDS:
            raise ValueError(
                f"Reference audio is {ref_seconds:.0f}s; the maximum is {runtime.MAX_REFERENCE_SECONDS:.0f}s. "
                "Breeze spends prompt budget at 12.5 tokens per second of reference, so a long clip "
                "leaves no room to speak. Trim the clip and provide its exact transcript."
            )

    if ref_codes is not None:
        cond = runtime.ref_segments(ref_text.strip(), text, instruction)
        negative = runtime.ref_segments(ref_text.strip(), text, instruction, with_instruction=False)
    else:
        cond = runtime.design_segments(text, instruction)
        negative = runtime.design_negative_segments(text)

    runtime.fix_seed(int(seed))
    inputs_embeds, attention_mask, base_positions, prefill_len = runtime.build_generation_batch(
        bundle.model,
        bundle.tokenizer,
        cond_segments=cond,
        negative_segments=negative if cfg_scale != 1.0 else None,
        ref_codes=ref_codes,
        cfg_scale=float(cfg_scale),
        device=device,
    )
    max_frames = min(int(max_new_tokens), runtime.MAX_SEQ_LEN - 1 - prefill_len)
    if max_frames < 64:
        raise ValueError(
            f"The prompt alone is {prefill_len} tokens of the {runtime.MAX_SEQ_LEN}-token context, "
            f"leaving only {max_frames} audio frames. Use a shorter reference clip or shorter text."
        )
    logger.info(
        "Prompt: %d tokens | up to %d frames (%.0fs of speech) | cfg %.1f",
        prefill_len, max_frames, max_frames / runtime.FRAMES_PER_SECOND, float(cfg_scale),
    )
    params = runtime.GenerationParams(
        max_new_tokens=max_frames,
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
        repetition_penalty=float(repetition_penalty),
        depth_temperature=float(depth_temperature),
        depth_top_k=int(depth_top_k),
        depth_top_p=float(depth_top_p),
    )

    est_frames = min(runtime.estimate_speech_frames(bundle.tokenizer, text), max_frames)
    logger.info(
        "Prompt: %d tokens | approx %ds of speech (%d frames, cap %d) | cfg %.1f",
        prefill_len, est_frames / runtime.FRAMES_PER_SECOND, est_frames, max_frames, float(cfg_scale),
    )

    pbar = ProgressBar(max_frames) if ProgressBar is not None else None
    cli_pbar = (
        tqdm(total=est_frames, desc="Breeze TTS 2 (~%.0fs)" % (est_frames / runtime.FRAMES_PER_SECOND),
             unit="frame", dynamic_ncols=True, leave=True)
        if tqdm is not None else None
    )

    def progress_callback(current: int) -> None:
        if pbar is not None:
            pbar.update_absolute(min(current, max_frames), max_frames)
        if cli_pbar is not None and current > cli_pbar.n:
            cli_pbar.update(current - cli_pbar.n)

    try:
        with torch.inference_mode(), native.attention_runtime(bundle.attention):
            codes = runtime.generate_codes(
                bundle.model,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                base_positions=base_positions,
                prefill_len=prefill_len,
                cfg_scale=float(cfg_scale),
                params=params,
                progress_callback=progress_callback,
            )
        wav = runtime.decode_codes(bundle.codec, codes)
    finally:
        if cli_pbar is not None:
            cli_pbar.total = cli_pbar.n
            cli_pbar.close()

    if not bool(torch.isfinite(wav).all()):
        raise RuntimeError("Breeze TTS 2 generated non-finite audio samples.")
    if wav.numel() == 0:
        raise RuntimeError("Breeze TTS 2 produced no audio.")
    return runtime.tensor_audio_to_comfy(wav)


def _stitch_reference(audio: dict, ref_audio: dict, mode: str) -> dict:
    if mode == "none":
        return audio
    ref_wav, ref_sr = runtime.comfy_audio_to_tensor(ref_audio)
    if ref_wav.numel() == 0:
        return audio
    sample_rate = int(audio["sample_rate"])
    if ref_sr != sample_rate:
        import torchaudio.functional as AF

        ref_wav = AF.resample(ref_wav, ref_sr, sample_rate)
    generated = audio["waveform"].view(-1)
    parts = (ref_wav, generated) if mode == "before" else (generated, ref_wav)
    stitched = torch.cat(parts).clamp(-1.0, 1.0)
    return {"waveform": stitched.view(1, 1, -1).contiguous(), "sample_rate": sample_rate}


class BreezeTTS2VoiceClone:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "breeze_model": ("BREEZE_TTS2_MODEL",),
            "text": _text_input(
                "(sigh) It is good to hear your voice again after all this time.",
                "Text to speak. Vocal events like (laugh) (sigh) (cough) (clears throat) work inline; use [笑] [叹气] etc. in Chinese.",
            ),
            "reference_audio": ("AUDIO", {"tooltip": "Clean reference speech to clone timbre, rhythm, and style from."}),
            "reference_text": _text_input(
                "This is the exact transcript of the reference audio.",
                "The exact transcript of the reference audio.",
            ),
            "cfg_scale": _cfg_input(1.0, "Guidance scale. 1.0 clones the reference as-is; raise it to push away from the reference."),
        }
        required.update(_generation_controls())
        return {"required": required}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "clone"
    CATEGORY = CATEGORY
    DESCRIPTION = "Clone a speaker from clean reference audio and its exact transcript."

    def clone(self, breeze_model, text, reference_audio, reference_text, cfg_scale, **controls):
        audio = _generate_audio(
            breeze_model,
            text=text,
            instruction=runtime.DEFAULT_INSTRUCTION,
            ref_audio=reference_audio,
            ref_text=reference_text,
            cfg_scale=cfg_scale,
            **controls,
        )
        return (audio,)


class BreezeTTS2VoiceDesign:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "breeze_model": ("BREEZE_TTS2_MODEL",),
            "text": _text_input(
                "(sigh) Welcome aboard. Your journey begins now.",
                "Text to speak. Vocal events like (laugh) (sigh) (cough) (clears throat) work inline; use [笑] [叹气] etc. in Chinese.",
            ),
            "instruction": _text_input(
                "A warm, thoughtful young woman with a clear voice and a calm, reflective delivery.",
                "Natural-language description of the voice to create. Match the instruction language to the text language.",
            ),
            "cfg_scale": _cfg_input(4.0, "Guidance scale. 4 is recommended for instruction-following."),
        }
        required.update(_generation_controls())
        return {"required": required}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "design"
    CATEGORY = CATEGORY
    DESCRIPTION = "Create a voice from a natural-language description, without reference audio."

    def design(self, breeze_model, text, instruction, cfg_scale, **controls):
        audio = _generate_audio(
            breeze_model,
            text=text,
            instruction=instruction,
            ref_audio=None,
            ref_text=None,
            cfg_scale=cfg_scale,
            **controls,
        )
        return (audio,)


class BreezeTTS2VoiceDirection:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "breeze_model": ("BREEZE_TTS2_MODEL",),
            "text": _text_input(
                "(clears throat) We need to discuss what happened last night.",
                "Text to speak. Vocal events like (laugh) (sigh) (cough) (clears throat) work inline; use [笑] [叹气] etc. in Chinese.",
            ),
            "reference_audio": ("AUDIO", {"tooltip": "Reference speech whose speaker identity is kept."}),
            "reference_text": _text_input(
                "This is the exact transcript of the reference audio.",
                "The exact transcript of the reference audio.",
            ),
            "instruction": _text_input(
                "Speak slowly with a restrained, serious tone.",
                "Direction for tone, emotion, pace, and delivery applied on top of the cloned voice.",
            ),
            "cfg_scale": _cfg_input(4.0, "Guidance scale. 4 is recommended for instruction-following."),
            "stitch_reference": (
                ["none", "before", "after"],
                {
                    "default": "none",
                    "tooltip": (
                        "Stitch the original reference clip into the output audio: "
                        "'none' returns the generated speech only, 'before' plays the reference clip first, "
                        "'after' appends it at the end. Purely an output edit; generation is unchanged."
                    ),
                },
            ),
        }
        required.update(_generation_controls())
        return {"required": required}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "direct"
    CATEGORY = CATEGORY
    DESCRIPTION = "Clone a voice from reference audio while steering tone, emotion, pace, and delivery."

    def direct(self, breeze_model, text, reference_audio, reference_text, instruction, cfg_scale, stitch_reference="none", **controls):
        audio = _generate_audio(
            breeze_model,
            text=text,
            instruction=instruction,
            ref_audio=reference_audio,
            ref_text=reference_text,
            cfg_scale=cfg_scale,
            **controls,
        )
        return (_stitch_reference(audio, reference_audio, stitch_reference),)


NODE_CLASS_MAPPINGS = {
    "BreezeTTS2LoadModel": BreezeTTS2LoadModel,
    "BreezeTTS2VoiceClone": BreezeTTS2VoiceClone,
    "BreezeTTS2VoiceDesign": BreezeTTS2VoiceDesign,
    "BreezeTTS2VoiceDirection": BreezeTTS2VoiceDirection,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BreezeTTS2LoadModel": "Breeze TTS 2 Load Model",
    "BreezeTTS2VoiceClone": "Breeze TTS 2 Voice Clone",
    "BreezeTTS2VoiceDesign": "Breeze TTS 2 Voice Design",
    "BreezeTTS2VoiceDirection": "Breeze TTS 2 Voice Direction",
}
