#!/usr/bin/env python3
"""
Inferencia de clonación de voz.
Soporta:
- F5-TTS (recomendado)
- XTTS-v2 (Coqui)
"""

import argparse
from pathlib import Path
import torch


def run_f5(ref_audio: str, text: str, output: str, model_path: str = None):
    """Inferencia con F5-TTS."""
    try:
        from f5_tts.api import F5TTS
    except ImportError:
        print("Instala f5-tts: pip install f5-tts")
        return

    print("Cargando F5-TTS...")
    # Si tienes checkpoint fine-tuned, pásalo aquí
    tts = F5TTS(model_type="F5-TTS", ckpt_file=model_path) if model_path else F5TTS()

    print(f"Generando: {text[:60]}...")
    wav, sr, _ = tts.infer(
        ref_file=ref_audio,
        ref_text="",          # puede dejarse vacío o poner la transcripción del ref
        gen_text=text,
        file_wave=output,
    )
    print(f"✅ Guardado en {output}")


def run_xtts(ref_audio: str, text: str, output: str, language: str = "es", model_path: str = None):
    """Inferencia con XTTS-v2."""
    try:
        from TTS.api import TTS
    except ImportError:
        print("Instala Coqui TTS: pip install TTS")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cargando XTTS-v2 en {device}...")

    if model_path:
        tts = TTS(model_path=model_path).to(device)
    else:
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    tts.tts_to_file(
        text=text,
        speaker_wav=ref_audio,
        language=language,
        file_path=output,
    )
    print(f"✅ Guardado en {output}")


def main():
    parser = argparse.ArgumentParser(description="Clonación de voz - Inferencia")
    parser.add_argument("--ref_audio", type=str, required=True, help="Audio de referencia (6-15s)")
    parser.add_argument("--text", type=str, required=True, help="Texto a sintetizar")
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--model", type=str, default="f5", choices=["f5", "xtts"])
    parser.add_argument("--model_path", type=str, default=None, help="Ruta a checkpoint fine-tuned")
    parser.add_argument("--language", type=str, default="es")
    args = parser.parse_args()

    if not Path(args.ref_audio).exists():
        print(f"Error: no existe {args.ref_audio}")
        return

    if args.model == "f5":
        run_f5(args.ref_audio, args.text, args.output, args.model_path)
    else:
        run_xtts(args.ref_audio, args.text, args.output, args.language, args.model_path)


if __name__ == "__main__":
    main()
