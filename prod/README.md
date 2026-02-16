# 🎮 Steam Review Classifier

Clasificador de sentimiento de reseñas de videojuegos de Steam usando **BERT** fine-tuneado.

## Descripción

Este proyecto implementa un modelo de clasificación binaria (positiva/negativa) de reseñas de videojuegos de Steam. Se realizó fine-tuning completo sobre el modelo `bert-base-uncased` de Hugging Face, entrenado con el dataset [Steam Reviews](https://www.kaggle.com/datasets/andrewmvd/steam-reviews) de Kaggle.

## Tecnologías

- **Modelo**: BERT (bert-base-uncased) con fine-tuning completo
- **Framework**: PyTorch + Hugging Face Transformers
- **Interfaz**: Streamlit
- **Dataset**: Steam Reviews (~50,000 reseñas balanceadas)

## Estructura del Proyecto

```
stem-review-classifier/
├── data/                          # Datos y artefactos del entrenamiento
│   ├── clean_reviews.csv          # Dataset limpio
│   ├── tensors/                   # Datasets tokenizados (.pt)
│   ├── model_save/                # Modelo guardado post-entrenamiento
│   └── results/                   # Métricas y gráficos de evaluación
├── dev/                           # Scripts de desarrollo (ejecución secuencial)
│   ├── 01_download_and_clean.py   # Descarga y limpieza del dataset
│   ├── 02_tokenize_and_dataset.py # Tokenización con BERT y creación del dataset PyTorch
│   ├── 03_train_model.py          # Fine-tuning del modelo BERT
│   ├── 04_evaluate_model.py       # Evaluación con métricas y gráficos
│   └── 05_export_model.py         # Exportación del modelo para producción
├── prod/                          # Aplicación de producción
│   ├── app.py                     # Aplicación Streamlit
│   ├── utils.py                   # Funciones auxiliares
│   ├── model_files/               # Modelo exportado (generado por 05_export)
│   ├── requirements.txt           # Dependencias
│   └── README.md                  # Este archivo
```

## Cómo Ejecutar

### 1. Entrenamiento (en `dev/`)

Ejecutar los scripts en orden desde la carpeta `dev/`:

```bash
cd dev

# Paso 1: Descargar y limpiar el dataset
python 01_download_and_clean.py

# Paso 2: Tokenizar y crear datasets PyTorch
python 02_tokenize_and_dataset.py

# Paso 3: Entrenar el modelo (fine-tuning de BERT)
python 03_train_model.py

# Paso 4: Evaluar el modelo
python 04_evaluate_model.py

# Paso 5: Exportar modelo para producción
python 05_export_model.py
```

### 2. Aplicación Web (en `prod/`)

#### 🖥️ Ejecución Local

```bash
cd prod

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

#### ☁️ Deploy a Streamlit Cloud

Para hacer deploy en Streamlit Cloud, sigue la guía completa en:
📄 [STREAMLIT_DEPLOY.md](../STREAMLIT_DEPLOY.md)

**Resumen rápido:**
1. Ve a [share.streamlit.io](https://share.streamlit.io/)
2. Conecta tu cuenta de GitHub
3. Selecciona este repositorio y la ruta `prod/app.py`
4. Click en "Deploy"

> **Nota:** Los archivos del modelo están en Git LFS y Streamlit Cloud los descargará automáticamente.


## Metodología

1. **Preprocesamiento**: Limpieza agresiva de texto (URLs, arte ASCII, caracteres especiales, spam) y balanceo de clases.
2. **Tokenización**: Usando `BertTokenizer.encode_plus()` con max_length=128, padding, truncation y attention masks.
3. **Fine-tuning**: Entrenamiento completo de todos los parámetros de `BertForSequenceClassification` con AdamW (lr=2e-5) y linear scheduler, 3 épocas, batch_size=32.
4. **Evaluación**: Accuracy, MCC (Matthews Correlation Coefficient), F1-score, matriz de confusión y pruebas cualitativas.

## Trabajo Práctico Integrador

**Materia**: Redes Neuronales Profundas  
**Universidad**: UTN - Facultad Regional Mendoza
