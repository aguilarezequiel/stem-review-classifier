# ==============================================================================
# 03 - Entrenamiento del Modelo (Fine-Tuning de BERT)
# ==============================================================================
# Este script carga los datasets tokenizados, crea los DataLoaders,
# carga BertForSequenceClassification y realiza el fine-tuning completo.
#
# Sigue exactamente el patrón del notebook BERT_Fine_Tuning.ipynb de clase:
# - BertForSequenceClassification con num_labels=2
# - AdamW optimizer con lr=2e-5
# - Linear schedule with warmup
# - Training loop con validación por época
# ==============================================================================

import os
import time
import datetime
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

# ---- Configuración ----
TENSORS_DIR = "../data/tensors/"
MODEL_SAVE_DIR = "../data/model_save/"
MODEL_NAME = 'bert-base-uncased'
BATCH_SIZE = 32       # Los autores de BERT recomiendan 16 o 32
EPOCHS = 3            # BERT recomienda entre 2 y 4 épocas
LEARNING_RATE = 2e-5  # Valor recomendado para fine-tuning
EPSILON = 1e-8        # Epsilon para AdamW
SEED = 42


def setup_device():
    """Configura el dispositivo (GPU o CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("GPU no disponible, usando CPU.")
    return device


def set_seed(seed):
    """Establece la semilla para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_datasets(tensors_dir):
    """Carga los TensorDatasets guardados."""
    train_dataset = torch.load(os.path.join(tensors_dir, "train_dataset.pt"), weights_only=False)
    val_dataset = torch.load(os.path.join(tensors_dir, "val_dataset.pt"), weights_only=False)
    print(f"Train: {len(train_dataset):,} muestras")
    print(f"Val:   {len(val_dataset):,} muestras")
    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, batch_size):
    """
    Crea los DataLoaders para entrenamiento y validación.
    - Entrenamiento: muestreo aleatorio (RandomSampler)
    - Validación: muestreo secuencial (SequentialSampler)
    """
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size
    )

    validation_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size
    )

    return train_dataloader, validation_dataloader


def load_model(model_name, device):
    """
    Carga BertForSequenceClassification, el modelo BERT preentrenado
    con una capa de clasificación lineal encima.
    """
    print(f"\nCargando modelo {model_name}...")
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,             # Clasificación binaria: positivo/negativo
        output_attentions=False,
        output_hidden_states=False,
    )
    model.to(device)

    # Mostrar estructura del modelo
    params = list(model.named_parameters())
    print(f"El modelo tiene {len(params)} grupos de parámetros.")
    print(f"\n==== Capa de Embedding ====")
    for p in params[0:5]:
        print(f"  {p[0]:<55} {str(tuple(p[1].size())):>12}")
    print(f"\n==== Primer Transformer ====")
    for p in params[5:21]:
        print(f"  {p[0]:<55} {str(tuple(p[1].size())):>12}")
    print(f"\n==== Capa de Salida (Clasificación) ====")
    for p in params[-4:]:
        print(f"  {p[0]:<55} {str(tuple(p[1].size())):>12}")

    return model


def flat_accuracy(preds, labels):
    """Calcula la accuracy comparando predicciones vs labels."""
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    """Formatea tiempo en segundos a hh:mm:ss."""
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train(model, train_dataloader, validation_dataloader, device, epochs, lr, eps):
    """
    Bucle de entrenamiento y validación.
    Sigue exactamente el patrón del notebook BERT_Fine_Tuning.ipynb de clase.
    """
    # Optimizer: AdamW con weight decay
    optimizer = AdamW(model.parameters(), lr=lr, eps=eps)

    # Scheduler: linear con warmup
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    training_stats = []
    total_t0 = time.time()

    for epoch_i in range(epochs):
        print(f'\n{"="*40}')
        print(f'  Época {epoch_i + 1} / {epochs}')
        print(f'{"="*40}')

        # ---- ENTRENAMIENTO ----
        print("\nEntrenando...")
        t0 = time.time()
        total_train_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print(f'  Lote {step:>5,} de {len(train_dataloader):>5,}.  Tiempo: {elapsed}')

            # Mover batch a GPU
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Limpiar gradientes
            model.zero_grad()

            # Forward pass
            result = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
                return_dict=True
            )

            loss = result.loss
            logits = result.logits

            total_train_loss += loss.item()

            # Backward pass
            loss.backward()

            # Gradient clipping (evita gradientes explosivos)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Actualizar pesos y scheduler
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)

        print(f"\n  Pérdida promedio de entrenamiento: {avg_train_loss:.4f}")
        print(f"  Tiempo de entrenamiento: {training_time}")

        # ---- VALIDACIÓN ----
        print("\nValidando...")
        t0 = time.time()
        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0

        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                result = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                    return_dict=True
                )

            loss = result.loss
            logits = result.logits

            total_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        validation_time = format_time(time.time() - t0)

        print(f"  Accuracy: {avg_val_accuracy:.4f}")
        print(f"  Pérdida de validación: {avg_val_loss:.4f}")
        print(f"  Validación tomó: {validation_time}")

        training_stats.append({
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        })

    total_time = format_time(time.time() - total_t0)
    print(f"\nEntrenamiento completo! Tiempo total: {total_time}")

    return training_stats


def save_model(model, model_name, output_dir):
    """
    Guarda el modelo y tokenizer usando save_pretrained().
    Después se pueden recargar con from_pretrained().
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nGuardando modelo en {output_dir}...")

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)

    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)

    print("Modelo y tokenizer guardados correctamente.")


def save_training_stats(training_stats, output_dir):
    """Guarda las estadísticas de entrenamiento como CSV."""
    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index('epoch')
    stats_path = os.path.join(output_dir, "training_stats.csv")
    df_stats.to_csv(stats_path)
    print(f"Estadísticas guardadas en: {stats_path}")
    print(df_stats.to_string())


# ---- Ejecución principal ----
if __name__ == "__main__":
    # Configuración
    device = setup_device()
    set_seed(SEED)

    # Cargar datos
    train_dataset, val_dataset = load_datasets(TENSORS_DIR)

    # Crear DataLoaders
    train_dataloader, validation_dataloader = create_dataloaders(
        train_dataset, val_dataset, BATCH_SIZE
    )

    # Cargar modelo
    model = load_model(MODEL_NAME, device)

    # Entrenar
    training_stats = train(
        model, train_dataloader, validation_dataloader,
        device, EPOCHS, LEARNING_RATE, EPSILON
    )

    # Guardar modelo y estadísticas
    save_model(model, MODEL_NAME, MODEL_SAVE_DIR)
    save_training_stats(training_stats, MODEL_SAVE_DIR)
