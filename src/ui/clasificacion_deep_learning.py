# src/ui/clasificacion_deep_learning.py

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
import io

# Configuración del logger
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def cargar_modelo_primary(model_path):
    """
    Carga el modelo primario de clasificación de imágenes desde la ruta especificada.
    Este modelo clasifica la imagen en 'masas', 'calcificaciones' o 'no_encontrado'.

    :param model_path: Ruta al directorio del modelo primario.
    :return: Pipeline de clasificación de imágenes o None si falla la carga.
    """
    if not os.path.exists(model_path):
        st.error(f"La ruta del modelo primario especificada no existe: {model_path}")
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
        classifier_primary = pipeline("image-classification", model=model, image_processor=image_processor,
                                      device=device)
        return classifier_primary
    except Exception as e:
        st.error(f"Error al cargar el modelo primario con transformers: {e}")
        return None


def cargar_modelo_secondary_masas(model_path):
    """
    Carga el modelo secundario para clasificar masas desde la ruta especificada.
    Este modelo clasifica la masa en 'benigna' o 'maligna'.

    :param model_path: Ruta al directorio del modelo secundario para masas.
    :return: Pipeline de clasificación de imágenes o None si falla la carga.
    """
    if not os.path.exists(model_path):
        st.error(f"La ruta del modelo secundario de masas especificada no existe: {model_path}")
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
        classifier_secondary_masas = pipeline("image-classification", model=model, image_processor=image_processor,
                                              device=device)
        return classifier_secondary_masas
    except Exception as e:
        st.error(f"Error al cargar el modelo secundario de masas con transformers: {e}")
        return None


def cargar_modelo_secondary_calcifi(model_path):
    """
    Carga el modelo secundario CALCI para clasificar calcificaciones desde la ruta especificada.
    Este modelo clasifica la calcificación en 'benigna', 'sospechosa' o 'maligna'.

    :param model_path: Ruta al directorio del modelo CALCI para calcificaciones.
    :return: Pipeline de clasificación de imágenes o None si falla la carga.
    """
    if not os.path.exists(model_path):
        st.error(f"La ruta del modelo CALCI especificada no existe: {model_path}")
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
        classifier_secondary_calcifi = pipeline("image-classification", model=model, image_processor=image_processor,
                                                device=device)
        return classifier_secondary_calcifi
    except Exception as e:
        st.error(f"Error al cargar el modelo CALCI con transformers: {e}")
        return None


def leer_dicom(dicom_file):
    """
    Lee un archivo DICOM y lo convierte a una imagen PIL Image.

    :param dicom_file: Archivo DICOM cargado por el usuario (Streamlit UploadedFile).
    :return: Imagen PIL Image en formato RGB o None si falla la conversión.
    """
    try:
        # Leer el archivo DICOM desde el objeto UploadedFile
        dicom = pydicom.dcmread(dicom_file)
        original_image = dicom.pixel_array

        # Aplicar VOI LUT con prefer_lut=True (priorizando LUT si está presente)
        img_windowed = apply_voi_lut(original_image, dicom, prefer_lut=True)

        # Manejar Photometric Interpretation si es MONOCHROME1 (invertir la imagen)
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

        return image

    except Exception as e:
        logger.error(f"Error al procesar el archivo DICOM: {e}")
        st.error(f"Error al procesar el archivo DICOM: {e}")
        return None


def leer_imagen(imagen_file):
    """
    Lee una imagen PNG o JPG y la convierte a una imagen PIL Image.

    :param imagen_file: Archivo PNG o JPG cargado por el usuario (Streamlit UploadedFile).
    :return: Imagen PIL Image en formato RGB o None si falla la conversión.
    """
    try:
        # Leer la imagen usando PIL
        image = Image.open(imagen_file).convert('RGB')

        # Verificar el tamaño y redimensionar si es necesario
        if image.size != (224, 224):
            image = image.resize((224, 224))
            st.write(f"Imagen redimensionada a (224, 224)")
        else:
            st.write(f"Imagen ya tiene el tamaño (224, 224)")

        return image
    except Exception as e:
        logger.error(f"Error al procesar la imagen: {e}")
        st.error(f"Error al procesar la imagen: {e}")
        return None


def procesar_archivo(uploaded_file):
    """
    Procesa un archivo de imagen en formato DICOM, PNG o JPG y lo convierte a una imagen PIL Image de 224x224 píxeles.

    :param uploaded_file: Archivo cargado por el usuario (Streamlit UploadedFile).
    :return: Tupla (imagen PIL Image, tipo de archivo) o (None, None) si falla la conversión.
    """
    try:
        filename = uploaded_file.name
        extension = os.path.splitext(filename)[1].lower()

        if extension in ['.dcm', '.dicom']:
            # Procesar archivo DICOM
            image = leer_dicom(uploaded_file)
            return image, 'DICOM'

        elif extension in ['.png', '.jpg', '.jpeg']:
            # Procesar archivo PNG o JPG
            image = leer_imagen(uploaded_file)
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
        mapped_results = {prediction_mapping.get(result['label'], result['label']): result['score'] for result in
                          results}
        return mapped_results
    except Exception as e:
        st.error(f"Ocurrió un error durante la clasificación: {e}")
        return None


def mostrar_resultados_primary(mapped_result):
    """
    Muestra los resultados de la clasificación primaria en la interfaz de Streamlit.

    :param mapped_result: Diccionario con etiquetas mapeadas y sus respectivas puntuaciones.
    """
    if mapped_result:
        st.write("### Resultado de la Clasificación Primaria")
        for label, score in mapped_result.items():
            st.write(f"**{label.capitalize()}**: {score * 100:.2f}%")
    else:
        st.write("No se pudieron obtener resultados de la clasificación primaria.")


def mostrar_resultados_secondary_masas(mapped_result):
    """
    Muestra los resultados de la clasificación secundaria para masas en la interfaz de Streamlit.

    :param mapped_result: Diccionario con etiquetas mapeadas y sus respectivas puntuaciones.
    """
    if mapped_result:
        st.write("### Resultado de la Clasificación Secundaria para Masas")
        for label, score in mapped_result.items():
            st.write(f"**{label.capitalize()}**: {score * 100:.2f}%")
    else:
        st.write("No se pudieron obtener resultados de la clasificación secundaria para masas.")


def mostrar_resultados_secondary_calcifi(mapped_result):
    """
    Muestra los resultados de la clasificación secundaria para calcificaciones en la interfaz de Streamlit.

    :param mapped_result: Diccionario con etiquetas mapeadas y sus respectivas puntuaciones.
    """
    if mapped_result:
        st.write("### Resultado de la Clasificación Secundaria para Calcificaciones")
        for label, score in mapped_result.items():
            st.write(f"**{label.capitalize()}**: {score * 100:.2f}%")
    else:
        st.write("No se pudieron obtener resultados de la clasificación secundaria para calcificaciones.")
