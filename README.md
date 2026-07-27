# Python Voice Cloning Intermediate 🎤

Proyecto de **nivel intermedio** de clonación de voz en Python, optimizado para **RTX 4070 Ti** (12 GB VRAM).

Este repositorio te guía paso a paso para:

1. Preparar un dataset de calidad a partir de audio propio
2. Hacer fine-tuning de modelos modernos (F5-TTS recomendado + XTTS-v2 como alternativa)
3. Generar audio de alta calidad con la voz entrenada
4. (Opcional) Crear una interfaz simple con Gradio

Ideal para quienes ya saben Python básico y quieren pasar del zero-shot simple al entrenamiento personalizado.

---

## Requisitos de hardware

| Componente     | Recomendado              | Mínimo                  |
|----------------|--------------------------|-------------------------|
| GPU            | RTX 4070 Ti (12 GB)      | RTX 3060 12 GB          |
| RAM            | 32 GB                    | 16 GB                   |
| Disco          | 50+ GB libres (SSD)      | 30 GB                   |
| CUDA           | 12.1+                    | 11.8+                   |

La 4070 Ti es perfecta: permite batch sizes decentes y entrenamiento relativamente rápido.

---

## Instalación rápida

```bash
# 1. Clonar el repositorio
git clone https://github.com/Keingranjefe/python-voice-cloning-intermediate.git
cd python-voice-cloning-intermediate

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate   # Linux/Mac
# o
venv\Scripts\activate      # Windows

# 3. Instalar PyTorch con CUDA (ajusta según tu CUDA)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Instalar dependencias del proyecto
pip install -r requirements.txt
```

### Instalar F5-TTS (modelo principal recomendado)

```bash
pip install f5-tts
# o desde el repo oficial para la última versión de fine-tuning:
# git clone https://github.com/SWivid/F5-TTS.git
# cd F5-TTS && pip install -e .
```

---

## Estructura del proyecto

```
python-voice-cloning-intermediate/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── default.yaml
├── src/
│   ├── dataset_prep.py      # Limpieza, segmentación y transcripción
│   ├── train_f5.py          # Fine-tuning de F5-TTS
│   ├── inference.py         # Generación de audio
│   └── utils.py
├── scripts/
│   ├── prepare_dataset.sh
│   └── run_finetune.sh
├── data/
│   ├── raw/                 # Audios originales (pon aquí tus .wav/.mp3)
│   ├── processed/           # Dataset listo para entrenamiento
│   └── README.md
└── models/                  # Checkpoints guardados aquí
```

---

## Flujo de trabajo (nivel intermedio)

### 1. Preparar el dataset (lo más importante)

Necesitas **15-60 minutos** de audio limpio de la voz que quieres clonar.

Recomendaciones:
- Audio mono, 24 kHz o 44.1 kHz
- Sin música de fondo ni ruido fuerte
- Segmentos de 3-15 segundos idealmente
- Transcripciones precisas

```bash
# Pon tus audios en data/raw/
python src/dataset_prep.py --input_dir data/raw --output_dir data/processed --language es
```

El script:
- Convierte a formato correcto
- Segmenta con silencios
- Transcribe con Whisper (large-v3 o turbo)
- Genera el metadata.csv necesario para F5-TTS / Coqui

### 2. Fine-tuning con F5-TTS

```bash
python src/train_f5.py --config config/default.yaml
```

Parámetros optimizados para 4070 Ti (12 GB):
- Batch size bajo + gradient accumulation
- Mixed precision (bf16 o fp16)
- Learning rate conservador

Tiempo estimado: 4-12 horas dependiendo de la cantidad de datos y steps.

### 3. Inferencia

```bash
python src/inference.py \
  --model_path models/f5_finetuned \
  --ref_audio data/processed/ref.wav \
  --text "Hola, esta es una prueba de mi voz clonada con fine-tuning." \
  --output output.wav
```

---

## Alternativa: XTTS-v2 (Coqui)

Si prefieres XTTS (más maduro en multi-idioma):

```bash
pip install TTS
# Luego usa los scripts de fine-tuning oficiales de Coqui o adapta train_f5.py
```

XTTS es excelente para cross-lingual (hablar otros idiomas con tu voz).

---

## Consejos de calidad

1. **Calidad del dataset > cantidad**  
   20 minutos limpios > 2 horas ruidosos.

2. Usa **Whisper large-v3** o **distil-whisper** para transcripciones precisas.

3. Normaliza volumen y elimina silencias largas.

4. Monitorea el loss y genera samples cada X steps.

5. Ética: Solo clona voces con permiso explícito.

---

## Roadmap del proyecto

- [x] Estructura base + README
- [ ] Script completo de preparación de dataset
- [ ] Fine-tuning F5-TTS optimizado para 12 GB
- [ ] Script de inferencia con Gradio
- [ ] Soporte para voice conversion (RVC) como post-proceso
- [ ] Pipeline de doblaje multi-idioma preservando prosodia

---

## Créditos y recursos

- [F5-TTS](https://github.com/SWivid/F5-TTS) - Modelo principal
- [Coqui TTS / XTTS](https://github.com/coqui-ai/TTS)
- [Whisper](https://github.com/openai/whisper)
- Comunidad LocalLLaMA y r/LocalLLaMA

---

**Autor:** Keingranjefe  
**GPU target:** RTX 4070 Ti  
**Nivel:** Intermedio  

¡Vamos a clonar voces de verdad! 🚀
