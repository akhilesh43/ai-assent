"""
Switch JARVIS to the LOCAL EAR listening mode and (optionally) tune it.

    python enable_local_ear.py                      # sensible defaults
    python enable_local_ear.py --speed fast         # quicker reply, less checking
    python enable_local_ear.py --model medium       # more accurate Whisper (needs more RAM/VRAM)
    python enable_local_ear.py --device cpu         # if the GPU gives errors
    python enable_local_ear.py --language hi        # pin Hindi (default: detect each time)
    python enable_local_ear.py --off                # back to Smart Turn

It only edits config/api_keys.json through memory.config_manager (never by hand), then
tells you what is missing. Restart JARVIS afterwards.
"""
import argparse
import sys
from importlib.util import find_spec

from memory.config_manager import (get_listening_mode, save_listening_mode,
                                   save_local_ear_config, get_local_ear_config)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--speed", choices=("fast", "accurate"))
    ap.add_argument("--model", choices=("tiny", "base", "small", "medium"))
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"))
    ap.add_argument("--language", help='"hi", "en", … or "" to detect every time')
    ap.add_argument("--denoise", type=float, help="0 (off) … 1 (strongest); default 0.5")
    ap.add_argument("--off", action="store_true", help="return to Smart Turn")
    a = ap.parse_args()

    if a.off:
        print("Listening mode ->", save_listening_mode("smart_vad"))
        return 0

    changes = {k: v for k, v in (("speed", a.speed), ("model", a.model), ("device", a.device),
                                 ("language", a.language), ("denoise", a.denoise)) if v is not None}
    cfg = save_local_ear_config(**changes) if changes else get_local_ear_config()
    mode = save_listening_mode("local_ear")

    print(f"Listening mode : {mode}")
    print(f"Speed          : {cfg['speed']}   (fast = quicker, accurate = every sentence checked)")
    print(f"Whisper        : model={cfg['model']} device={cfg['device']} language={cfg['language'] or 'auto'}")
    print(f"Denoise        : {cfg['denoise']}   Song mode: {'ON (say Hello)' if cfg['media_wake'] else 'off'}")

    problems = []
    if find_spec("faster_whisper") is None:
        problems.append("faster-whisper is not installed -> pip install faster-whisper\n"
                        "      (needed for 'accurate' mode and for Hello-while-a-song-plays)")
    if find_spec("scipy") is None:
        print("note: scipy not installed (optional) -> pip install scipy")
    if cfg["device"] in ("auto", "cuda") and find_spec("ctranslate2") is not None:
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() == 0:
                print("note: no CUDA GPU seen by Whisper — it will run on the CPU (fine, a bit slower).")
        except Exception:
            pass
    elif cfg["device"] in ("auto", "cuda"):
        print("note: for an NVIDIA GPU also: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12")
    if problems:
        print("\nTO DO:")
        for p in problems:
            print("  -", p)
    print("\nDone. Restart JARVIS. (Press the 🎧 LISTEN button to cycle modes any time.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
