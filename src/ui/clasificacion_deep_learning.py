# src/ui/clasificacion_deep_learning.py

import streamlit as st
from PIL import Image
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
import os
from transformers import pipeline
import torch
import cv2
import logging
import io

# Configuración del logger
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def cargar_modelo(model_path):
    """
    Carga el modelo de clasificación de imágenes desde la ruta especificada.
    Utiliza GPU si está disponible, de lo contrario, usa CPU o MPS en dispositivos Apple.

    :param model_path: Ruta al directorio del modelo.
    :return: Pipeline de clasificación de imágenes o None si falla la carga.
    """
    if not os.path.exists(model_path):
        st.error(f"La ruta del modelo especificada no existe: {model_path}")
        return None

    try:
        if torch.cuda.is_available():
            device = 0  # Usar GPU CUDA
        elif torch.backends.mps.is_available():
            device = "mps"  # Usar GPU Apple MPS
        else:
            device = -1  # Usar CPU

        classifier = pipeline("image-classification", model=model_path, device=device)
        return classifier
    except Exception as e:
        st.error(f"Ocurrió un error al cargar el modelo: {e}")
        return None


def convertir_dicom_a_imagen(dicom_file, output_size=(224, 224)):
    """
    Convierte un archivo DICOM a una imagen PIL Image con el tamaño especificado.

    :param dicom_file: Archivo DICOM cargado por el usuario (Streamlit UploadedFile).
    :param output_size: Tupla (ancho, alto) para redimensionar la imagen.
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
        image = image.resize(output_size)

        return image

    except Exception as e:
        logger.error(f"Error al procesar el archivo DICOM: {e}")
        st.error(f"Error al procesar el archivo DICOM: {e}")
        return None


def convertir_png_jpg_a_png(imagen_file, output_size=(224, 224)):
    """
    Convierte un archivo PNG o JPG a una imagen PIL Image de tamaño 224x224 píxeles.
    Si la imagen ya es PNG y de tamaño 224x224, no la convierte.

    :param imagen_file: Archivo PNG o JPG cargado por el usuario (Streamlit UploadedFile).
    :param output_size: Tupla (ancho, alto) para redimensionar la imagen.
    :return: Imagen PIL Image en formato PNG y tamaño 224x224 o None si falla la conversión.
    """
    try:
        # Leer la imagen usando PIL
        image = Image.open(imagen_file).convert('RGB')

        # Verificar si la imagen ya está en PNG
        extension = os.path.splitext(imagen_file.name)[1].lower()
        es_png = extension == '.png'

        # Verificar el tamaño y redimensionar si es necesario
        if image.size != output_size:
            image = image.resize(output_size)
            st.write(f"Imagen redimensionada a {output_size}")
        else:
            st.write(f"Imagen ya tiene el tamaño {output_size}")

        # Si la imagen original no es PNG, convertirla a PNG
        if not es_png:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            image = Image.open(img_byte_arr)

        return image

    except Exception as e:
        logger.error(f"Error al procesar la imagen: {e}")
        st.error(f"Error al procesar la imagen: {e}")
        return None


def procesar_archivo(imagen_file):
    """
    Procesa un archivo de imagen en formato DICOM, PNG o JPG y lo convierte a PNG de 224x224 píxeles según sea necesario.

    :param imagen_file: Archivo cargado por el usuario (Streamlit UploadedFile).
    :return: Tupla (imagen PIL Image, tipo de archivo) o (None, None) si falla la conversión.
    """
    try:
        # Obtener el nombre del archivo y su extensión
        filename = imagen_file.name
        extension = os.path.splitext(filename)[1].lower()

        if extension in ['.dcm', '.dicom']:
            # Procesar archivo DICOM
            image = convertir_dicom_a_imagen(imagen_file, output_size=(224, 224))
            return image, 'DICOM'

        elif extension in ['.png', '.jpg', '.jpeg']:
            # Procesar archivo PNG o JPG
            image = convertir_png_jpg_a_png(imagen_file, output_size=(224, 224))
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
    :return: Lista de diccionarios con etiquetas mapeadas y sus respectivas puntuaciones.
    """
    try:
        resultado = classifier(image)
        mapped_result = [
            {
                'label': prediction_mapping.get(item['label'], item['label']),
                'score': item['score']
            }
            for item in resultado
        ]
        return mapped_result
    except Exception as e:
        st.error(f"Ocurrió un error durante la clasificación: {e}")
        return None


def mostrar_resultados(mapped_result):
    """
    Muestra los resultados de la clasificación en la interfaz de Streamlit.

    :param mapped_result: Lista de diccionarios con etiquetas mapeadas y sus respectivas puntuaciones.
    """
    if mapped_result:
        st.write("### Resultado de la Clasificación")
        for res in mapped_result:
            st.write(f"**{res['label'].capitalize()}**: {res['score'] * 100:.2f}%")
    else:
        st.write("No se pudieron obtener resultados de la clasificación.")
