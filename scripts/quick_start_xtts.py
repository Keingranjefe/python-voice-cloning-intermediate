#!/usr/bin/env python3
"""
Ejemplo rápido de zero-shot con XTTS-v2.
Útil para probar que todo funciona antes del fine-tuning.
"""

from TTS.api import TTS
import torch

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    # Cambia esta ruta por un audio de referencia tuyo (6-15 segundos limpios)
    ref_audio = "data/raw/tu_voz_ejemplo.wav"
    text = "Hola, esta es una prueba de clonación de voz con XTTS. Sueno bastante natural, ¿verdad?"

    output = "output_xtts_zero_shot.wav"

    print("Generando audio...")
    tts.tts_to_file(
        text=text,
        speaker_wav=ref_audio,
        language="es",
        file_path=output,
    )
    print(f"✅ Listo → {output}")

if __name__ == "__main__":
    main()
