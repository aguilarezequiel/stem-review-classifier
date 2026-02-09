# ==============================================================================
# 04 - Evaluación del Modelo en el Conjunto de Test
# ==============================================================================
# Este script carga el modelo fine-tuneado y lo evalúa sobre el conjunto
# de test. Calcula métricas como accuracy, precision, recall, F1-score,
# matriz de confusión y MCC (Matthews Correlation Coefficient).
#
# Sigue el patrón de evaluación del notebook BERT_Fine_Tuning.ipynb.
# ==============================================================================

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    accuracy_score
)
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para guardar gráficos
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Configuración ----
TENSORS_DIR = "../data/tensors/"
MODEL_DIR = "../data/model_save/"
RESULTS_DIR = "../data/results/"
BATCH_SIZE = 32


def setup_device():
    """Configura el dispositivo (GPU o CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Usando CPU.")
    return device


def load_test_data(tensors_dir):
    """Carga el dataset de test."""
    test_dataset = torch.load(
        os.path.join(tensors_dir, "test_dataset.pt"), weights_only=False
    )
    print(f"Test: {len(test_dataset):,} muestras")
    return test_dataset


def predict(model, test_dataloader, device):
    """
    Genera predicciones sobre el conjunto de test.
    Sigue el patrón del notebook de clase.
    """
    print(f"\nPrediciendo etiquetas para {len(test_dataloader.dataset):,} muestras...")
    model.eval()

    predictions = []
    true_labels = []

    for batch in test_dataloader:
        # Mover batch a GPU
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Forward sin calcular gradientes
        with torch.no_grad():
            result = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                return_dict=True
            )

        logits = result.logits
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.append(logits)
        true_labels.append(label_ids)

    # Combinar resultados de todos los batches
    flat_predictions = np.concatenate(predictions, axis=0)
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = np.concatenate(true_labels, axis=0)

    print("Predicción completa.")
    return flat_predictions, flat_true_labels


def compute_metrics(predictions, true_labels):
    """Calcula y muestra todas las métricas de evaluación."""
    print("\n" + "=" * 50)
    print("  RESULTADOS DE EVALUACIÓN")
    print("=" * 50)

    # Accuracy
    acc = accuracy_score(true_labels, predictions)
    print(f"\nAccuracy: {acc:.4f}")

    # MCC (Matthews Correlation Coefficient)
    mcc = matthews_corrcoef(true_labels, predictions)
    print(f"MCC: {mcc:.4f}")

    # Classification Report
    target_names = ['Negativa', 'Positiva']
    report = classification_report(true_labels, predictions, target_names=target_names)
    print(f"\nClassification Report:\n{report}")

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    print(f"Confusion Matrix:\n{cm}")

    return acc, mcc, report, cm


def plot_confusion_matrix(cm, output_dir):
    """Genera y guarda el gráfico de la matriz de confusión."""
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Negativa', 'Positiva'],
        yticklabels=['Negativa', 'Positiva']
    )
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción')
    plt.title('Matriz de Confusión - Steam Review Classifier')
    plt.tight_layout()

    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    print(f"\nMatriz de confusión guardada en: {path}")
    plt.close()


def plot_training_stats(model_dir, output_dir):
    """Grafica las curvas de pérdida de entrenamiento y validación."""
    stats_path = os.path.join(model_dir, "training_stats.csv")
    if not os.path.exists(stats_path):
        print("No se encontró training_stats.csv, saltando gráfico.")
        return

    df_stats = pd.read_csv(stats_path)

    sns.set(style='darkgrid')
    sns.set(font_scale=1.5)
    plt.figure(figsize=(12, 6))

    plt.plot(df_stats['epoch'], df_stats['Training Loss'], 'b-o', label="Entrenamiento")
    plt.plot(df_stats['epoch'], df_stats['Valid. Loss'], 'g-o', label="Validación")

    plt.title("Pérdida de Entrenamiento y Validación")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.legend()
    plt.xticks(df_stats['epoch'])
    plt.tight_layout()

    path = os.path.join(output_dir, "training_loss.png")
    plt.savefig(path, dpi=150)
    print(f"Curva de pérdida guardada en: {path}")
    plt.close()


def test_with_examples(model, tokenizer, device):
    """
    Prueba el modelo con reseñas de ejemplo para verificar cualitativamente.
    """
    examples = [
        "This game is absolutely amazing, I love the graphics and gameplay!",
        "Terrible game, crashes every 5 minutes. Do not buy this garbage.",
        "It's okay, nothing special but not bad either. Average game.",
        "Best game I have ever played. Hundreds of hours of fun!",
        "Waste of money. The developers abandoned this project.",
        "Really fun with friends, great multiplayer experience.",
    ]

    print("\n" + "=" * 50)
    print("  PRUEBAS CUALITATIVAS")
    print("=" * 50)

    model.eval()
    for text in examples:
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        with torch.no_grad():
            result = model(input_ids, attention_mask=attention_mask, return_dict=True)

        logits = result.logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

        label = "POSITIVA" if pred == 1 else "NEGATIVA"
        print(f"\n  [{label}] (conf: {confidence:.2%})")
        print(f"  \"{text[:80]}...\"" if len(text) > 80 else f"  \"{text}\"")


# ---- Ejecución principal ----
if __name__ == "__main__":
    import pandas as pd

    device = setup_device()

    # Cargar modelo fine-tuneado
    print(f"Cargando modelo desde {MODEL_DIR}...")
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model.to(device)

    # Cargar datos de test
    test_dataset = load_test_data(TENSORS_DIR)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=BATCH_SIZE
    )

    # Generar predicciones
    predictions, true_labels = predict(model, test_dataloader, device)

    # Calcular métricas
    acc, mcc, report, cm = compute_metrics(predictions, true_labels)

    # Guardar gráficos
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_confusion_matrix(cm, RESULTS_DIR)
    plot_training_stats(MODEL_DIR, RESULTS_DIR)

    # Pruebas cualitativas
    test_with_examples(model, tokenizer, device)

    # Guardar métricas en archivo
    with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"MCC: {mcc:.4f}\n\n")
        f.write(f"Classification Report:\n{report}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")

    print(f"\nResultados guardados en: {RESULTS_DIR}")
