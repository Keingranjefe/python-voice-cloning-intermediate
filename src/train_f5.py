#!/usr/bin/env python3
"""
Launcher de fine-tuning para F5-TTS.
Optimizado para RTX 4070 Ti (12 GB).

Nota: F5-TTS tiene su propio sistema de entrenamiento.
Este script prepara el entorno y lanza el comando oficial
con hiperparámetros seguros para 12 GB de VRAM.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import yaml


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune F5-TTS")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--dry_run", action="store_true", help="Solo mostrar el comando")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(cfg["data"]["train_data_dir"])
    metadata = Path(cfg["data"]["metadata_file"])

    if not metadata.exists():
        print(f"❌ No se encontró {metadata}")
        print("Ejecuta primero: python src/dataset_prep.py")
        sys.exit(1)

    print("=" * 60)
    print("Fine-tuning F5-TTS - Optimizado para RTX 4070 Ti")
    print("=" * 60)
    print(f"Dataset:     {data_dir}")
    print(f"Metadata:    {metadata}")
    print(f"Output:      {cfg['project']['output_dir']}")
    print(f"Batch size:  {cfg['training']['batch_size_per_gpu']}")
    print(f"Grad accum:  {cfg['training']['grad_accumulation_steps']}")
    print(f"LR:          {cfg['training']['learning_rate']}")
    print(f"Max steps:   {cfg['training']['max_steps']}")
    print("=" * 60)

    # Comando típico de F5-TTS (ajusta según la versión instalada)
    # Revisa: https://github.com/SWivid/F5-TTS/tree/main/src/f5_tts/train
    cmd = [
        sys.executable, "-m", "f5_tts.train",
        # Los argumentos exactos dependen de la versión de F5-TTS.
        # Ejemplo común (verifica con --help):
        # "--dataset_name", str(data_dir),
        # "--exp_name", cfg["project"]["name"],
        # etc.
    ]

    print("\n⚠️  IMPORTANTE:")
    print("F5-TTS tiene un sistema de entrenamiento propio.")
    print("Recomendación actual (2026):")
    print()
    print("1. Instala la versión más reciente:")
    print("   pip install f5-tts")
    print("   # o clona el repo y haz pip install -e .")
    print()
    print("2. Prepara el dataset en el formato que espera F5-TTS")
    print("   (normalmente un CSV con path|text)")
    print()
    print("3. Usa el script oficial de fine-tuning:")
    print("   f5-tts_finetune-gradio   # interfaz gráfica fácil")
    print("   # o el comando de train desde el repo")
    print()
    print("Parámetros recomendados para 4070 Ti (12 GB):")
    print(f"  - batch_size_per_gpu: {cfg['training']['batch_size_per_gpu']}")
    print(f"  - grad_accumulation_steps: {cfg['training']['grad_accumulation_steps']}")
    print(f"  - learning_rate: {cfg['training']['learning_rate']}")
    print(f"  - mixed_precision: {cfg['training']['mixed_precision']}")
    print()
    print("Una vez que tengas el checkpoint fine-tuned, úsalo con:")
    print("  python src/inference.py --model f5 --model_path models/... --ref_audio ... --text ...")
    print()

    if args.dry_run:
        print("(dry-run: no se ejecutó nada)")
        return

    # Por ahora solo informamos. Cuando el usuario confirme el formato exacto
    # de la versión de F5-TTS instalada, se puede completar el launcher.
    print("Script listo como guía. Ejecuta el fine-tuning con la herramienta oficial de F5-TTS")
    print("y apunta el checkpoint resultante a este proyecto.")


if __name__ == "__main__":
    main()
