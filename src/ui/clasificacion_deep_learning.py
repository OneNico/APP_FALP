# src/ui/clasificacion_deep_learning.py

from transformers import pipeline, AutoConfig, AutoModelForImageClassification
import torch
import streamlit as st
import os

def cargar_modelo(model_path):
    try:
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForImageClassification.from_pretrained(model_path, trust_remote_code=True)
        classifier = pipeline("image-classification", model=model, config=config)
        return classifier
    except Exception as e:
        st.error(f"Error al cargar el modelo con transformers: {e}")
        return None

def cargar_modelo_pytorch(model_path):
    try:
        # Ajusta el nombre del archivo de pesos según corresponda
        model_file = os.path.join(model_path, 'model.safetensors')  # Cambiado a 'model.safetensors'
        if not os.path.exists(model_file):
            st.error(f"Archivo de modelo no encontrado: {model_file}")
            return None

        # Cargar el modelo usando transformers que soporta safetensors
        model = torch.jit.load(model_file, map_location='cpu')
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo con PyTorch: {e}")
        return None

def procesar_archivo(uploaded_file):
    import pydicom
    from PIL import Image
    import io
    import numpy as np

    try:
        if uploaded_file.name.endswith(('.dcm', '.dicom')):
            dicom = pydicom.dcmread(uploaded_file)
            image = dicom.pixel_array
            # Normalizar y convertir a PIL Image
            image = Image.fromarray(image).convert('RGB')
            image = image.resize((224, 224))
            return image, 'dicom'
        else:
            image = Image.open(uploaded_file).convert('RGB')
            image = image.resize((224, 224))
            return image, 'image'
    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
        return None, None

def clasificar_imagen(image, classifier, prediction_mapping):
    try:
        results = classifier(image)
        # Mapear etiquetas
        mapped_results = {prediction_mapping.get(result['label'], result['label']): result['score'] for result in results}
        return mapped_results
    except Exception as e:
        st.error(f"Error al clasificar la imagen: {e}")
        return None

def mostrar_resultados(mapped_result):
    if mapped_result:
        st.write("### Resultados de la Clasificación:")
        for label, score in mapped_result.items():
            st.write(f"**{label}:** {score:.4f}")
