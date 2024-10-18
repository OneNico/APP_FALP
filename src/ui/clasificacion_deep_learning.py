# src/ui/clasificacion_deep_learning.pyy

import streamlit as st
from PIL import Image
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
import os
from transformers import pipeline, AutoImageProcessor, AutoConfig, AutoModelForImageClassification
import torch
from safetensors.torch import load_file  # Asegúrate de tener safetensors instalado
import logging

# Configuración del logger
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

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
        config = AutoConfig.from_pretrained(model_path)
        image_processor = AutoImageProcessor.from_pretrained(model_path)
        
        # Cargar modelo
        model = AutoModelForImageClassification.from_pretrained(model_path, trust_remote_code=True)
        
        # Determinar dispositivo
        if torch.cuda.is_available():
            device = 0  # GPU CUDA
        elif torch.backends.mps.is_available():
            device = "mps"  # GPU Apple MPS
        else:
            device = -1  # CPU

        # Crear pipeline
        classifier = pipeline("image-classification", model=model, image_processor=image_processor, device=device)
        return classifier
    except Exception as e:
        st.error(f"Error al cargar el modelo con transformers: {e}")
        return None

def procesar_archivo(uploaded_file):
    """
    Procesa un archivo de imagen en formato DICOM, PNG o JPG y lo convierte a PIL Image de 224x224 píxeles.

    :param uploaded_file: Archivo cargado por el usuario (Streamlit UploadedFile).
    :return: Tupla (imagen PIL Image, tipo de archivo) o (None, None) si falla la conversión.
    """
    try:
        filename = uploaded_file.name
        extension = os.path.splitext(filename)[1].lower()

        if extension in ['.dcm', '.dicom']:
            # Procesar archivo DICOM
            dicom = pydicom.dcmread(uploaded_file)
            original_image = dicom.pixel_array

            # Aplicar VOI LUT con prefer_lut=True
            img_windowed = apply_voi_lut(original_image, dicom, prefer_lut=True)

            # Manejar Photometric Interpretation
            photometric_interpretation = dicom.get('PhotometricInterpretation', 'UNKNOWN')
            if photometric_interpretation == 'MONOCHROME1':
                img_windowed = img_windowed.max() - img_windowed
                st.write(f"Imagen invertida debido a Photometric Interpretation: {photometric_interpretation}")
            else:
                st.write(f"Photometric Interpretation: {photometric_interpretation}")

            # Normalizar la imagen para que esté en el rango [0, 255]
            img_normalized = (img_windowed - img_windowed.min()) / (img_windowed.max() - img_windowed.min()) * 255
            img_normalized = img_normalized.astype(np.uint8)

            # Convertir a PIL Image
            image = Image.fromarray(img_normalized).convert('RGB')

            # Redimensionar a 224x224
            image = image.resize((224, 224))

            return image, 'DICOM'

        elif extension in ['.png', '.jpg', '.jpeg']:
            # Procesar archivo PNG o JPG
            image = Image.open(uploaded_file).convert('RGB')

            # Verificar si la imagen ya está en PNG
            es_png = extension == '.png'

            # Verificar el tamaño y redimensionar si es necesario
            if image.size != (224, 224):
                image = image.resize((224, 224))
                st.write(f"Imagen redimensionada a (224, 224)")
            else:
                st.write(f"Imagen ya tiene el tamaño (224, 224)")

            # Si la imagen original no es PNG, convertirla a PNG
            if not es_png:
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                image = Image.open(img_byte_arr)

            return image, 'PNG_JPG'

        else:
            st.error("Formato de archivo no soportado. Por favor, carga una imagen en formato DICOM, PNG o JPG.")
            return None, None

    except Exception as e:
        logger.error(f"Error al procesar el archivo: {e}")
        st.error(f"Error al procesar el archivo: {e}")
        return None, None

def clasificar_imagen(image, classifier, prediction_mapping):
    """
    Realiza la inferencia sobre una imagen y mapea las etiquetas predichas.

    :param image: Imagen PIL Image a clasificar.
    :param classifier: Pipeline de clasificación de imágenes.
    :param prediction_mapping: Diccionario para mapear etiquetas predichas a etiquetas legibles.
    :return: Diccionario con etiquetas mapeadas y sus respectivas puntuaciones.
    """
    try:
        results = classifier(image)
        # Mapear etiquetas
        mapped_results = {prediction_mapping.get(result['label'], result['label']): result['score'] for result in results}
        return mapped_results
    except Exception as e:
        st.error(f"Ocurrió un error durante la clasificación: {e}")
        return None

def mostrar_resultados(mapped_result):
    """
    Muestra los resultados de la clasificación en la interfaz de Streamlit.

    :param mapped_result: Diccionario con etiquetas mapeadas y sus respectivas puntuaciones.
    """
    if mapped_result:
        st.write("### Resultado de la Clasificación")
        for label, score in mapped_result.items():
            st.write(f"**{label.capitalize()}**: {score * 100:.2f}%")
    else:
        st.write("No se pudieron obtener resultados de la clasificación.")
