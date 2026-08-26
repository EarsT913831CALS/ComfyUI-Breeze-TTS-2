"""ComfyUI-Breeze-TTS-2: voice clone, voice design, and voice direction with Breeze TTS 2."""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import sys
import types

__version__ = "v1.1.0"

logger = logging.getLogger("BreezeTTS2")
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[BreezeTTS2] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _block_broken_torchcodec() -> None:
    """Stub out a broken torchcodec install so transformers audio imports survive."""
    try:
        torchcodec = importlib.import_module("torchcodec")
        if torchcodec is not None and getattr(torchcodec, "__spec__", None) is not None:
            return
        raise ImportError("torchcodec is broken")
    except Exception:
        pass

    stub = types.ModuleType("torchcodec")
    stub.__spec__ = importlib.machinery.ModuleSpec("torchcodec", None)
    stub.__version__ = "0.0.0"
    decoders = types.ModuleType("torchcodec.decoders")

    class _Missing:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torchcodec is stubbed out by ComfyUI-Breeze-TTS-2.")

    stub.AudioDecoder = _Missing
    decoders.AudioDecoder = _Missing
    stub.decoders = decoders
    sys.modules.setdefault("torchcodec", stub)
    sys.modules.setdefault("torchcodec.decoders", decoders)

    original_version = importlib.metadata.version

    def patched_version(name):
        if name == "torchcodec":
            return "0.0.0"
        return original_version(name)

    importlib.metadata.version = patched_version


_block_broken_torchcodec()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .loader import register_model_folder

    register_model_folder()

    from .nodes import NODE_CLASS_MAPPINGS as _NODES, NODE_DISPLAY_NAME_MAPPINGS as _NODE_NAMES
    from .whisper import NODE_CLASS_MAPPINGS as _WHISPER, NODE_DISPLAY_NAME_MAPPINGS as _WHISPER_NAMES

    NODE_CLASS_MAPPINGS.update(_NODES)
    NODE_CLASS_MAPPINGS.update(_WHISPER)
    NODE_DISPLAY_NAME_MAPPINGS.update(_NODE_NAMES)
    NODE_DISPLAY_NAME_MAPPINGS.update(_WHISPER_NAMES)
    logger.info("Registered %d node(s).", len(NODE_CLASS_MAPPINGS))
except Exception:
    logger.exception("Failed to register Breeze TTS 2 nodes.")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
