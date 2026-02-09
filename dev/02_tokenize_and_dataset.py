# ==============================================================================
# 02 - Tokenización y Creación del Dataset en PyTorch
# ==============================================================================
# Este script toma el CSV limpio generado en el paso anterior, tokeniza
# las reseñas usando el tokenizer de BERT (bert-base-uncased), y crea
# los TensorDatasets para entrenamiento, validación y test.
#
# Sigue el patrón del notebook BERT_Fine_Tuning.ipynb de clase:
# - tokenizer() para tokenizar (equivalente moderno de encode_plus)
# - TensorDataset con (input_ids, attention_masks, labels)
# - Split 80/10/10 (train/val/test)
# ==============================================================================

import os
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split

# ---- Configuración ----
CLEAN_DATA_PATH = "../data/clean_reviews.csv"
TENSORS_DIR = "../data/tensors/"
MAX_LENGTH = 128              # Longitud máxima de tokens (BERT soporta hasta 512)
MODEL_NAME = 'bert-base-uncased'
RANDOM_SEED = 42


def load_clean_data(path):
    """Carga el CSV limpio con columnas 'text' y 'label'."""
    df = pd.read_csv(path)
    print(f"Dataset cargado: {len(df):,} muestras")
    print(f"Distribución: {df['label'].value_counts().to_dict()}")
    return df


def find_max_length(sentences, tokenizer):
    """
    Encuentra la longitud máxima de tokens en el dataset.
    Útil para decidir el MAX_LENGTH de padding.
    """
    max_len = 0
    for sent in sentences:
        input_ids = tokenizer.encode(sent, add_special_tokens=True)
        max_len = max(max_len, len(input_ids))
    print(f"Longitud máxima encontrada: {max_len} tokens")
    return max_len


def tokenize_sentences(sentences, labels_array, tokenizer, max_length):
    """
    Tokeniza todas las oraciones usando tokenizer.encode_plus().
    Retorna tensores de input_ids, attention_masks y labels.

    Este proceso sigue exactamente el patrón del notebook de clase.
    """
    input_ids = []
    attention_masks = []

    print(f"Tokenizando {len(sentences):,} reseñas...")

    for i, sent in enumerate(sentences):
        if (i + 1) % 5000 == 0:
            print(f"  Procesadas {i+1:,} / {len(sentences):,}")

        # encode_plus realiza:
        #   (1) Tokeniza la oración
        #   (2) Agrega [CLS] al inicio y [SEP] al final
        #   (3) Mapea tokens a sus IDs
        #   (4) Pad/truncate a max_length
        #   (5) Crea la máscara de atención para los tokens [PAD]
        encoded_dict = tokenizer(
            sent,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Convertir listas a tensores
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels_array, dtype=torch.long)

    print(f"Tokenización completa.")
    print(f"  input_ids shape: {input_ids.shape}")
    print(f"  attention_masks shape: {attention_masks.shape}")
    print(f"  labels shape: {labels.shape}")

    return input_ids, attention_masks, labels


def split_dataset(input_ids, attention_masks, labels, seed):
    """
    Divide el dataset en train (80%), validation (10%) y test (10%).
    Usa TensorDataset y random_split como en el notebook de clase.
    """
    dataset = TensorDataset(input_ids, attention_masks, labels)

    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    print(f"\nDivisión del dataset:")
    print(f"  Entrenamiento: {train_size:,} muestras")
    print(f"  Validación:    {val_size:,} muestras")
    print(f"  Test:          {test_size:,} muestras")

    return train_dataset, val_dataset, test_dataset


def save_csv_splits(df, train_dataset, val_dataset, test_dataset, output_dir):
    """
    Guarda los splits como CSV en la carpeta data/ (requerido por el TP).
    Usa los índices del random_split para mapear a las filas originales.
    """
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    test_indices = test_dataset.indices

    df_train = df.iloc[train_indices].reset_index(drop=True)
    df_val = df.iloc[val_indices].reset_index(drop=True)
    df_test = df.iloc[test_indices].reset_index(drop=True)

    df_train.to_csv(os.path.join(output_dir, "training_data.csv"), index=False)
    df_val.to_csv(os.path.join(output_dir, "validation_data.csv"), index=False)
    df_test.to_csv(os.path.join(output_dir, "test_data.csv"), index=False)

    print(f"\nCSVs guardados en {output_dir}:")
    print(f"  training_data.csv   ({len(df_train):,} filas)")
    print(f"  validation_data.csv ({len(df_val):,} filas)")
    print(f"  test_data.csv       ({len(df_test):,} filas)")


def save_datasets(train_dataset, val_dataset, test_dataset, output_dir):
    """Guarda los datasets como archivos .pt para cargar después."""
    os.makedirs(output_dir, exist_ok=True)

    torch.save(train_dataset, os.path.join(output_dir, "train_dataset.pt"))
    torch.save(val_dataset, os.path.join(output_dir, "val_dataset.pt"))
    torch.save(test_dataset, os.path.join(output_dir, "test_dataset.pt"))

    print(f"\nDatasets guardados en: {output_dir}")


def show_tokenization_example(sentences, tokenizer, max_length):
    """Muestra un ejemplo de tokenización para verificar el proceso."""
    example = sentences[0]
    print(f"\n--- Ejemplo de tokenización ---")
    print(f"Original:  {example[:100]}...")
    print(f"Tokenized: {tokenizer.tokenize(example)[:20]}...")

    encoded = tokenizer(
        example,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    print(f"Token IDs: {encoded['input_ids'][0][:20]}...")
    print(f"Att. Mask: {encoded['attention_mask'][0][:20]}...")


# ---- Ejecución principal ----
if __name__ == "__main__":
    # Cargar datos limpios
    df = load_clean_data(CLEAN_DATA_PATH)
    sentences = df['text'].values
    labels = df['label'].values

    # Cargar el tokenizer de BERT
    print(f"\nCargando tokenizer de BERT ({MODEL_NAME})...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=True)

    # Mostrar un ejemplo de tokenización
    show_tokenization_example(sentences, tokenizer, MAX_LENGTH)

    # Tokenizar todo el dataset
    input_ids, attention_masks, labels_tensor = tokenize_sentences(
        sentences, labels, tokenizer, MAX_LENGTH
    )

    # Dividir en train/val/test
    train_dataset, val_dataset, test_dataset = split_dataset(
        input_ids, attention_masks, labels_tensor, RANDOM_SEED
    )

    # Guardar los datasets como tensores (.pt)
    save_datasets(train_dataset, val_dataset, test_dataset, TENSORS_DIR)

    # Guardar los splits como CSV en data/ (requerido por el TP)
    data_dir = os.path.dirname(CLEAN_DATA_PATH)
    save_csv_splits(df, train_dataset, val_dataset, test_dataset, data_dir)

    print("\nProceso de tokenización y creación del dataset completado.")
