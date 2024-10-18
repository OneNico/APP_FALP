# src/utilidades/importar_modelo.py

import os
import logging
import time
import random
import zipfile
import gdown
import streamlit as st
from transformers import AutoModelForImageClassification, AutoImageProcessor, pipeline
import torch

# Configuración del logger
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

@st.cache_resource
def descargar_modelo(model_dir, model_folder, file_id):
    """
    Descarga y extrae el modelo desde Google Drive si no existe localmente.
    
    :param model_dir: Directorio donde se almacenarán los modelos.
    :param model_folder: Nombre de la carpeta del modelo.
    :param file_id: ID del archivo ZIP en Google Drive.
    :return: Ruta al modelo o None si falla la descarga/extracción.
    """
    model_path = os.path.join(model_dir, model_folder)
    
    # Asegurar que el directorio model_dir existe
    os.makedirs(model_dir, exist_ok=True)
    
    if os.path.exists(model_path):
        st.info("El modelo ya está presente localmente.")
        return model_path

    st.info("Descargando el modelo desde Google Drive. Esto puede tardar unos minutos...")

    # Crear una barra de progreso
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        zip_path = os.path.join(model_dir, f"{model_folder}.zip")

        # Simular progreso de descarga (barra de progreso real con gdown no es directamente posible)
        for percent_complete in range(1, 101):
            progress_bar.progress(percent_complete)
            status_text.text(f"Descargando... {percent_complete}%")
            time.sleep(random.uniform(0.005, 0.02))  # Simular tiempo de descarga

        # Descargar el archivo real usando gdown
        gdown.download(url, zip_path, quiet=True)

        # Extraer el archivo ZIP
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        
        os.remove(zip_path)
        st.success("Modelo descargado y extraído correctamente.")
        return model_path
    except Exception as e:
        logger.error(f"Error al descargar o extraer el modelo: {e}")
        st.error(f"Error al descargar o extraer el modelo: {e}")
        return None

def cargar_modelo(model_path):
    """
    Carga el modelo de clasificación de imágenes desde la ruta especificada.
    Utiliza GPU si está disponible, de lo contrario, usa CPU.

    :param model_path: Ruta al directorio del modelo.
    :return: Pipeline de clasificación de imágenes o None si falla la carga.
    """
    if not os.path.exists(model_path):
        st.error(f"La ruta del modelo especificada no existe: {model_path}")
        return None

    try:
        # Cargar configuración y procesador de imágenes
        image_processor = AutoImageProcessor.from_pretrained(model_path)
        
        # Cargar modelo utilizando safetensors
        model = AutoModelForImageClassification.from_pretrained(
            model_path,
            trust_remote_code=True,  # Solo si tu modelo requiere código remoto
            from_safetensors=True    # Indica que usará safetensors
        )
        
        # Determinar dispositivo
        if torch.cuda.is_available():
            device = 0  # GPU CUDA
        elif torch.backends.mps.is_available():
            device = "mps"  # GPU Apple MPS
        else:
            device = -1  # CPU

        # Crear pipeline
        classifier = pipeline(
            "image-classification",
            model=model,
            image_processor=image_processor,
            device=device
        )
        
        return classifier
    except Exception as e:
        st.error(f"Error al cargar el modelo con transformers: {e}")
        return None
