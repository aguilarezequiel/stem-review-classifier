# ==============================================================================
# 05 - Exportar Modelo para Producción
# ==============================================================================
# Este script toma el modelo guardado con save_pretrained() y lo exporta
# en formato .pth (solo los pesos del state_dict) para uso en producción.
# También copia el tokenizer necesario.
#
# El archivo modelo.pth se usa en la app de Streamlit (carpeta prod/).
# ==============================================================================

import os
import torch
from transformers import BertForSequenceClassification, BertTokenizer

# ---- Configuración ----
MODEL_DIR = "../data/model_save/"
PROD_DIR = "../prod/"


def export_model(model_dir, prod_dir):
    """
    Exporta el modelo en formato .pth y copia los archivos del tokenizer.
    """
    print(f"Cargando modelo desde {model_dir}...")
    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)

    os.makedirs(prod_dir, exist_ok=True)

    # Guardar solo el state_dict (más liviano para producción)
    pth_path = os.path.join(prod_dir, "modelo.pth")
    torch.save(model.state_dict(), pth_path)
    file_size = os.path.getsize(pth_path) / (1024 * 1024)
    print(f"Modelo guardado: {pth_path} ({file_size:.1f} MB)")

    # También guardar con save_pretrained para facilitar la carga
    model_prod_dir = os.path.join(prod_dir, "model_files")
    os.makedirs(model_prod_dir, exist_ok=True)
    model.save_pretrained(model_prod_dir)
    tokenizer.save_pretrained(model_prod_dir)
    print(f"Modelo y tokenizer copiados a: {model_prod_dir}")

    # Verificar que se puede cargar correctamente
    print("\nVerificando carga del modelo...")
    loaded_model = BertForSequenceClassification.from_pretrained(model_prod_dir)
    loaded_tokenizer = BertTokenizer.from_pretrained(model_prod_dir)
    print(f"Modelo cargado correctamente: {type(loaded_model).__name__}")
    print(f"Tokenizer cargado correctamente. Vocab size: {loaded_tokenizer.vocab_size}")

    print("\nExportación completada. El modelo está listo para producción.")


if __name__ == "__main__":
    export_model(MODEL_DIR, PROD_DIR)
