# Steam Review Classifier 🎮

Aplicación de análisis de sentimientos sobre reseñas de videojuegos, construida sobre un modelo BERT (`bert-base-uncased`) fine-tuneado con el dataset de Steam Reviews de Kaggle.

Trabajo Práctico Integrador — Redes Neuronales Profundas · UTN Facultad Regional Mendoza

---

## ¿Qué hace?

Dado el texto de una reseña de Steam, el modelo predice si la opinión es **positiva** o **negativa** con un índice de confianza.

La aplicación web tiene dos modos:

- **Manual:** pegás cualquier texto y lo clasifica al instante.
- **Por App ID:** ingresás el ID de un juego de Steam, el sistema trae reseñas reales desde la API pública de Steam y las clasifica todas, mostrando un resumen de positivas/negativas y comparando contra las etiquetas originales de Steam.

**Accuracy sobre el conjunto de test: 90.6% (F1 macro: 0.91)**

---

## Estructura del proyecto

```
stem-review-classifier/
│
├── dev/                        # Pipeline de entrenamiento (scripts Python)
│   ├── 01_download_and_clean.py    # Descarga y limpieza del dataset de Kaggle
│   ├── 02_tokenize_and_dataset.py  # Tokenización con BERT y creación de tensores
│   ├── 03_train_model.py           # Fine-tuning de BERT (3 épocas, AdamW, lr=2e-5)
│   ├── 04_evaluate_model.py        # Evaluación sobre test set (accuracy, F1, MCC)
│   ├── 05_export_model.py          # Exporta el modelo entrenado a prod/model_files/
│   └── requirements.txt            # Dependencias del pipeline
│
├── notebooks/                  # Versión notebook de cada paso del pipeline
│   ├── 01_download_and_clean.ipynb
│   ├── 02_tokenize_and_dataset.ipynb
│   ├── 03_train_model.ipynb
│   ├── 04_evaluate_model.ipynb
│   └── 05_export_model.ipynb
│
├── prod/                       # Aplicación web lista para desplegar
│   ├── app.py                      # App Streamlit (interfaz completa)
│   ├── utils.py                    # Funciones: preprocesamiento, predicción, API de Steam
│   ├── model_files/                # Modelo y tokenizer exportados (safetensors + config)
│   └── requirements.txt            # Dependencias de producción
│
├── data/                       # Datos generados por el pipeline (no versionados en git)
│   ├── dataset.csv                 # Dataset original de Kaggle
│   ├── clean_reviews.csv           # Dataset limpio y balanceado (50k reseñas)
│   ├── training_data.csv           # Split de entrenamiento
│   ├── validation_data.csv         # Split de validación
│   ├── test_data.csv               # Split de test
│   ├── tensors/                    # TensorDatasets para PyTorch
│   ├── model_save/                 # Checkpoint del modelo tras el entrenamiento
│   └── results/                    # Métricas y gráficos de evaluación
│
└── docs/                       # Documentación del proyecto
    ├── Trabajo Práctico Integrador - Red Neuronal con Aplicación Web.pdf
    └── Presentacion_Final_Steam_Classifier.pptx
```

---

## Cómo ejecutar la app web

> Solo necesitás la carpeta `prod/`. El modelo ya está incluido en `prod/model_files/`.

```bash
cd prod
pip install -r requirements.txt
streamlit run app.py
```

La app queda disponible en `http://localhost:8501`.

---

## Cómo reproducir el entrenamiento

Si querés volver a entrenar el modelo desde cero, ejecutás los scripts de `dev/` en orden. Requiere tener Python y CUDA (GPU) disponibles para tiempos razonables.

### 1. Instalar dependencias

```bash
cd dev
pip install -r requirements.txt
```

### 2. Descargar el dataset de Kaggle

Necesitás tener `kaggle.json` configurado en `~/.kaggle/`.

```bash
kaggle datasets download -d andrewmvd/steam-reviews
# Descomprimí el archivo y colocá el CSV en data/dataset.csv
```

### 3. Ejecutar el pipeline

```bash
# Desde la carpeta dev/
python 01_download_and_clean.py      # Limpia y balancea el dataset (50k muestras)
python 02_tokenize_and_dataset.py    # Tokeniza y genera los TensorDatasets
python 03_train_model.py             # Fine-tuning de BERT (3 épocas, ~2-3h con GPU)
python 04_evaluate_model.py          # Evalúa sobre test set y guarda métricas
python 05_export_model.py            # Copia el modelo entrenado a prod/model_files/
```

Cada script puede ejecutarse también como notebook desde `notebooks/`.

---

## Tecnologías

| Componente | Tecnología |
|---|---|
| Modelo base | `bert-base-uncased` (Hugging Face Transformers) |
| Framework de entrenamiento | PyTorch 2.5 |
| Interfaz web | Streamlit 1.41 |
| Dataset | [Steam Reviews — Kaggle (andrewmvd)](https://www.kaggle.com/datasets/andrewmvd/steam-reviews) |
| API de reseñas en vivo | Steam Store API (pública) |

---

## Resultados

El modelo fue evaluado sobre un conjunto de test de **5.000 reseñas** balanceado (50% positivas / 50% negativas):

| Métrica | Valor |
|---|---|
| Accuracy | **90.6%** |
| F1-score (macro) | **0.91** |
| MCC | **0.81** |
| Precision (positiva) | 0.90 |
| Recall (positiva) | 0.91 |
