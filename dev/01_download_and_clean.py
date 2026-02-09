# ==============================================================================
# 01 - Descarga y Limpieza del Dataset de Steam Reviews
# ==============================================================================
# Este script descarga el dataset "Steam Reviews" de Kaggle (andrewmvd),
# realiza una limpieza agresiva del texto de las reseñas y guarda un CSV
# limpio y balanceado en la carpeta data/.
#
# Dataset: https://www.kaggle.com/datasets/andrewmvd/steam-reviews
# Columnas relevantes: review_text (texto), review_score (1=positivo, -1=negativo)
#
# Requisito previo: tener configurado `kaggle.json` en ~/.kaggle/
#   pip install kaggle
#   kaggle datasets download -d andrewmvd/steam-reviews
# ==============================================================================

import os
import re
import pandas as pd
import numpy as np

# ---- Configuración ----
RAW_DATA_PATH = "../data/dataset.csv"       # Ruta al CSV descargado de Kaggle
CLEAN_DATA_PATH = "../data/clean_reviews.csv"
SAMPLE_SIZE = 50000                          # Muestras totales (25K positivas + 25K negativas)
MIN_WORD_COUNT = 5                           # Mínimo de palabras tras la limpieza
MAX_WORD_COUNT = 200                         # Máximo de palabras (evita reseñas extremadamente largas)
RANDOM_SEED = 42


def load_raw_data(path):
    """
    Carga el CSV crudo del dataset de Kaggle.
    El dataset tiene múltiples archivos CSV; usamos el primero disponible.
    """
    # El dataset de andrewmvd puede venir como múltiples CSVs (dataset_1.csv, dataset_2.csv, etc.)
    # Si el path apunta a un archivo directo, lo cargamos así.
    # Si no, buscamos los archivos en la carpeta data/
    if os.path.isfile(path):
        print(f"Cargando datos desde {path}...")
        df = pd.read_csv(path)
    else:
        # Buscar archivos CSV en la carpeta data/
        data_dir = os.path.dirname(path)
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and 'clean' not in f]
        if not csv_files:
            raise FileNotFoundError(
                f"No se encontraron archivos CSV en {data_dir}.\n"
                "Descargá el dataset de Kaggle con:\n"
                "  kaggle datasets download -d andrewmvd/steam-reviews\n"
                "y descomprimí los archivos en la carpeta data/"
            )
        print(f"Archivos encontrados: {csv_files}")
        # Cargar y concatenar todos los CSVs
        dfs = []
        for f in csv_files:
            filepath = os.path.join(data_dir, f)
            print(f"  Leyendo {f}...")
            dfs.append(pd.read_csv(filepath))
        df = pd.concat(dfs, ignore_index=True)

    print(f"Datos cargados: {df.shape[0]:,} filas, {df.shape[1]} columnas")
    print(f"Columnas: {list(df.columns)}")
    return df


def clean_text(text):
    """
    Limpia agresivamente el texto de una reseña de Steam.
    Elimina: arte ASCII, caracteres especiales, URLs, exceso de puntuación, etc.
    """
    if not isinstance(text, str):
        return ""

    # Eliminar URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Eliminar bloques de arte ASCII (caracteres como ░▒▓█ y similares)
    text = re.sub(r'[░▒▓█▄▀■□▪▫●○◆◇♠♣♥♦♪♫☆★►◄▲▼←→↑↓]+', '', text)

    # Eliminar asteriscos usados como emotes (*joins server*)
    text = re.sub(r'\*[^*]+\*', '', text)

    # Eliminar viñetas/bullets (•, ►, ▪, etc.)
    text = re.sub(r'[•►▪▸‣⁃]', '', text)

    # Reemplazar fracciones y ratios comunes (10/10, 5/5, etc.) con palabras
    text = re.sub(r'(\d+)\s*/\s*(\d+)', r'\1 out of \2', text)

    # Eliminar resoluciones de pantalla y especificaciones técnicas
    text = re.sub(r'\d{3,4}\s*x\s*\d{3,4}', '', text)

    # Eliminar caracteres no ASCII excepto letras acentuadas básicas
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Eliminar números sueltos (pero mantener números pegados a palabras como "3D")
    text = re.sub(r'\b\d+\b', '', text)

    # Eliminar puntuación excesiva (más de 2 caracteres repetidos)
    text = re.sub(r'([!?.]){3,}', r'\1\1', text)

    # Eliminar caracteres especiales, manteniendo solo letras, espacios y puntuación básica
    text = re.sub(r"[^a-zA-Z\s.,!?'-]", ' ', text)

    # Colapsar múltiples espacios en uno solo
    text = re.sub(r'\s+', ' ', text)

    # Eliminar espacios al inicio y final
    text = text.strip()

    # Convertir a minúsculas (BERT uncased lo requiere)
    text = text.lower()

    return text


def is_valid_review(text, min_words=MIN_WORD_COUNT, max_words=MAX_WORD_COUNT):
    """
    Verifica si una reseña limpia es válida para entrenamiento.
    Filtra reseñas muy cortas, muy largas, o sin contenido real.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return False

    words = text.split()
    word_count = len(words)

    # Filtrar por cantidad de palabras
    if word_count < min_words or word_count > max_words:
        return False

    # Filtrar si más del 50% son caracteres repetidos (spam)
    unique_chars = len(set(text.replace(' ', '')))
    if unique_chars < 5:
        return False

    return True


def process_and_save(df, output_path, sample_size, seed):
    """
    Procesa el DataFrame: limpia texto, filtra reseñas inválidas,
    balancea clases y guarda el resultado.
    """
    print("\n--- Limpieza de texto ---")

    # Identificar columnas (el dataset puede tener nombres variados)
    # Intentar encontrar la columna de texto y la de score
    text_col = None
    score_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'review_text' in col_lower or 'review' == col_lower:
            text_col = col
        if 'review_score' in col_lower or 'recommendation' in col_lower:
            score_col = col

    if text_col is None or score_col is None:
        print(f"Columnas disponibles: {list(df.columns)}")
        # Fallback: asumir que las columnas relevantes son las típicas
        if 'review_text' in df.columns:
            text_col = 'review_text'
        if 'review_score' in df.columns:
            score_col = 'review_score'

    print(f"Columna de texto: {text_col}")
    print(f"Columna de score: {score_col}")

    # Eliminar filas con texto nulo
    df = df.dropna(subset=[text_col])
    print(f"Filas después de eliminar nulos: {df.shape[0]:,}")

    # Limpiar el texto
    print("Limpiando texto (esto puede tardar unos minutos)...")
    df['clean_text'] = df[text_col].apply(clean_text)

    # Filtrar reseñas inválidas
    df['is_valid'] = df['clean_text'].apply(is_valid_review)
    df = df[df['is_valid']].copy()
    print(f"Filas después de filtrar inválidas: {df.shape[0]:,}")

    # Mapear labels: review_score 1 -> 1 (positivo), -1 -> 0 (negativo)
    # (algunos datasets usan 1/-1, otros "Recommended"/"Not Recommended")
    if df[score_col].dtype == object:
        df['label'] = (df[score_col].str.lower() == 'recommended').astype(int)
    else:
        df['label'] = (df[score_col] > 0).astype(int)

    print(f"\nDistribución de clases:")
    print(f"  Positivas: {(df['label'] == 1).sum():,}")
    print(f"  Negativas: {(df['label'] == 0).sum():,}")

    # Balancear clases: tomar la misma cantidad de cada una
    half_sample = sample_size // 2
    pos = df[df['label'] == 1]
    neg = df[df['label'] == 0]

    n_pos = min(half_sample, len(pos))
    n_neg = min(half_sample, len(neg))
    n_each = min(n_pos, n_neg)  # Asegurar balance perfecto

    print(f"\nMuestreando {n_each:,} de cada clase (total: {n_each*2:,})")

    pos_sample = pos.sample(n=n_each, random_state=seed)
    neg_sample = neg.sample(n=n_each, random_state=seed)

    balanced_df = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=seed)

    # Guardar solo las columnas necesarias
    result = balanced_df[['clean_text', 'label']].reset_index(drop=True)
    result.columns = ['text', 'label']

    result.to_csv(output_path, index=False)
    print(f"\nDataset limpio guardado en: {output_path}")
    print(f"Total de muestras: {len(result):,}")
    print(f"Distribución final: {result['label'].value_counts().to_dict()}")

    # Mostrar ejemplos
    print("\n--- Ejemplos de reseñas limpias ---")
    for i, row in result.head(5).iterrows():
        label_str = "POSITIVA" if row['label'] == 1 else "NEGATIVA"
        print(f"\n[{label_str}] {row['text'][:150]}...")

    return result


# ---- Ejecución principal ----
if __name__ == "__main__":
    df = load_raw_data(RAW_DATA_PATH)
    process_and_save(df, CLEAN_DATA_PATH, SAMPLE_SIZE, RANDOM_SEED)
