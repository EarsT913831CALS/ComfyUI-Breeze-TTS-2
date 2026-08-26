"""Install helper for the Breeze TTS 2 node pack.

Never installs, upgrades, or removes torch / torchaudio / transformers / numpy
or any other heavyweight ComfyUI runtime package. Only installs missing
lightweight Python dependencies.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

CRITICAL_IMPORTS = ["torch", "torchaudio", "transformers", "numpy"]
LIGHTWEIGHT_IMPORTS = ["accelerate", "safetensors", "soundfile", "huggingface_hub", "tqdm"]


def _missing(names: list[str]) -> list[str]:
    return [name for name in names if importlib.util.find_spec(name) is None]


def main() -> int:
    missing_critical = _missing(CRITICAL_IMPORTS)
    if missing_critical:
        print(
            "Breeze TTS 2: missing critical packages: "
            + ", ".join(missing_critical)
            + ". Install this nodepack inside a working ComfyUI environment; "
            "this helper will not modify torch, torchaudio, or transformers."
        )
        return 1

    missing = _missing(LIGHTWEIGHT_IMPORTS)
    if not missing:
        print("Breeze TTS 2: all dependencies already available.")
        return 0

    print("Breeze TTS 2: installing missing packages:", ", ".join(missing))
    return subprocess.call([sys.executable, "-m", "pip", "install", *missing])


if __name__ == "__main__":
    raise SystemExit(main())
