#!/usr/bin/env python3
"""
Dataset preparation for intermediate voice cloning.
- Carga audios de data/raw
- Convierte a mono 24kHz
- Segmenta por silencios
- Transcribe con Whisper
- Genera metadata.csv compatible con F5-TTS / Coqui
"""

import argparse
import os
from pathlib import Path
import soundfile as sf
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import whisper


def load_and_resample(path: Path, target_sr: int = 24000):
    audio, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio, target_sr


def split_on_silence(audio, sr, min_silence_len=0.5, silence_thresh=-40, min_segment=2.0, max_segment=15.0):
    """Segmentación simple por energía."""
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    energy = np.array([
        np.sum(np.abs(audio[i:i+frame_length]**2))
        for i in range(0, len(audio)-frame_length, hop_length)
    ])
    energy_db = 10 * np.log10(energy + 1e-10)
    threshold = np.percentile(energy_db, 20)  # aproximado

    # Encontrar regiones de habla
    is_speech = energy_db > (threshold + 5)
    segments = []
    start = None
    for i, speech in enumerate(is_speech):
        time = i * hop_length / sr
        if speech and start is None:
            start = time
        elif not speech and start is not None:
            end = time
            if end - start >= min_segment:
                segments.append((start, min(end, start + max_segment)))
            start = None
    if start is not None:
        end = len(audio) / sr
        if end - start >= min_segment:
            segments.append((start, min(end, start + max_segment)))
    return segments


def main():
    parser = argparse.ArgumentParser(description="Preparar dataset de voz")
    parser.add_argument("--input_dir", type=str, default="data/raw")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--language", type=str, default="es", help="Idioma para Whisper")
    parser.add_argument("--whisper_model", type=str, default="large-v3", help="tiny, base, small, medium, large-v3, turbo")
    parser.add_argument("--sample_rate", type=int, default=24000)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(exist_ok=True)

    print(f"Cargando Whisper ({args.whisper_model})...")
    model = whisper.load_model(args.whisper_model)

    audio_files = list(input_dir.glob("*.wav")) + list(input_dir.glob("*.mp3")) + list(input_dir.glob("*.flac"))
    if not audio_files:
        print(f"No se encontraron audios en {input_dir}")
        return

    records = []
    global_idx = 0

    for audio_path in tqdm(audio_files, desc="Procesando audios"):
        try:
            audio, sr = load_and_resample(audio_path, args.sample_rate)
            segments = split_on_silence(audio, sr)

            if not segments:
                # Si no hay segmentos, usar todo el archivo (si es corto)
                duration = len(audio) / sr
                if 2.0 <= duration <= 30.0:
                    segments = [(0.0, duration)]

            for start, end in segments:
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                segment = audio[start_sample:end_sample]

                # Transcribir
                # Whisper espera 16kHz internamente, pero acepta el array
                result = model.transcribe(segment, language=args.language, fp16=True)
                text = result["text"].strip()

                if len(text) < 3:
                    continue

                filename = f"{global_idx:05d}.wav"
                out_path = wavs_dir / filename
                sf.write(out_path, segment, sr)

                # Formato metadata común: filename|text  o  path|text|speaker
                records.append({
                    "file_name": f"wavs/{filename}",
                    "text": text,
                    "speaker": "target_voice"
                })
                global_idx += 1

        except Exception as e:
            print(f"Error procesando {audio_path}: {e}")

    if records:
        df = pd.DataFrame(records)
        metadata_path = output_dir / "metadata.csv"
        # Formato simple para F5-TTS / muchos pipelines
        df.to_csv(metadata_path, index=False, sep="|", header=False)
        # También guardar versión con header para inspección
        df.to_csv(output_dir / "metadata_with_header.csv", index=False)
        print(f"\n✅ Dataset listo: {len(records)} segmentos")
        print(f"   Metadata: {metadata_path}")
        print(f"   Audios:   {wavs_dir}")
    else:
        print("No se generaron segmentos válidos.")


if __name__ == "__main__":
    main()
