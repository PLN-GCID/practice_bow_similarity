# Práctica: Similitud de documentos con Bag-of-Words y TF-IDF

Cálculo de la similitud entre documentos (tecnología móvil y vehículos
eléctricos) usando distintas representaciones vectoriales y la métrica de
similitud del coseno, implementado con PyTorch.

## Estructura

- `data/` — enunciado (`Práctica.pdf`) y los 12 documentos fuente (`doc01.docx`...`doc12.docx`).
- `bow/` — parte 1: representación Bag-of-Words.
  - `bow_similarity.py` — preprocesamiento, construcción del BoW y similitud del coseno.
  - `informe_bow.md` — hallazgos.
- `tfidf/` — parte 2: representación TF-IDF.
  - `tfidf_similarity.py` — ponderación TF-IDF (reutiliza el preprocesamiento
    y la matriz BoW de la parte 1), similitud del coseno y comparación con BoW.
  - `informe_tfidf.md` — hallazgos y comparación con el modelo BoW.

## Uso

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch python-docx numpy

python bow/bow_similarity.py
python tfidf/tfidf_similarity.py
```
