# Datos del proyecto

## data/raw/
Coloca aquí tus audios originales de la voz que quieres clonar.

**Recomendaciones:**
- Formato: `.wav`, `.mp3` o `.flac`
- Duración total: 15-60 minutos (más es mejor si es limpio)
- Calidad: poco ruido de fondo, sin música, volumen consistente
- Ideal: grabaciones de voz hablada natural (podcasts, lecturas, conversaciones)

## data/processed/
Aquí se genera el dataset después de ejecutar:

```bash
python src/dataset_prep.py --input_dir data/raw --output_dir data/processed --language es
```

Contenido esperado:
- `wavs/` → segmentos de audio (2-15 segundos)
- `metadata.csv` → formato `path|texto`
- `metadata_with_header.csv` → para revisión humana

## Consejo de calidad

Antes de entrenar, escucha 10-20 segmentos al azar y corrige manualmente las transcripciones malas.  
Una buena transcripción es más importante que tener muchos minutos de audio.
