# app.py

import streamlit as st
from src.ui.visualizacion import mostrar_visualizacion
from src.ui.convertir_png import mostrar_convertir_png  # Importar la nueva función
from src.ui.clasificacion_deep_learning import (
    cargar_modelo_primary,
    cargar_modelo_secondary_masas,
    cargar_modelo_secondary_calcifi,
    procesar_archivo,
    clasificar_imagen,
    mostrar_resultados_primary,
    mostrar_resultados_secondary_masas,
    mostrar_resultados_secondary_calcifi
)
from PIL import Image
import os
import torch
import requests
import zipfile
import logging
import time
import random

# Configuración del logger
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


@st.cache_resource
def descargar_modelo(model_dir, model_folder, file_url):
    """
    Descarga y extrae el modelo desde SharePoint si no existe localmente.
    """
    model_path = os.path.join(model_dir, model_folder)
    os.makedirs(model_dir, exist_ok=True)

    if os.path.exists(model_path):
        st.info(f"El modelo '{model_folder}' ya está presente localmente.")
        return model_path

    st.info(f"Descargando el modelo '{model_folder}' desde SharePoint. Esto puede tardar unos minutos...")

    try:
        # Obtener el contenido del archivo ZIP desde el enlace de SharePoint
        with st.spinner(f"Descargando {model_folder}..."):
            response = requests.get(file_url, stream=True)
            if response.status_code != 200:
                st.error(f"Error al descargar el modelo '{model_folder}'. Código de estado: {response.status_code}")
                return None

            zip_path = os.path.join(model_dir, f"{model_folder}.zip")
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        # Extraer el archivo ZIP
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)

        # Eliminar el archivo ZIP descargado
        os.remove(zip_path)
        st.success(f"Modelo '{model_folder}' descargado y extraído correctamente.")
        return model_path
    except Exception as e:
        logger.error(f"Error al descargar o extraer el modelo '{model_folder}': {e}")
        st.error(f"Error al descargar o extraer el modelo '{model_folder}': {e}")
        return None


def main():
    # Cargar el archivo CSS externo
    def cargar_css():
        try:
            with open('styles/style.css') as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        except FileNotFoundError:
            st.warning("Archivo CSS no encontrado. Usando estilos por defecto.")

    cargar_css()

    # HTML para el título dentro de la estructura proporcionada
    title_html = """
    <div class="outer">
        <div class="dot"></div>
        <div class="card">
            <div class="ray"></div>
            <div class="text">Clasificación de Mamografías</div>
            <div class="line topl"></div>
            <div class="line leftl"></div>
            <div class="line bottoml"></div>
            <div class="line rightl"></div>
        </div>
    </div>
    """
    st.markdown(title_html, unsafe_allow_html=True)

    # Configurar la barra lateral
    st.sidebar.header("Opciones de Procesamiento")
    tipo_carga = st.sidebar.radio(
        "Selecciona el tipo de carga",
        ["Procesamiento de DICOM", "Clasificación mediante Deep Learning"]
    )

    opciones = {'tipo_carga': tipo_carga}

    if tipo_carga == "Procesamiento de DICOM":
        st.sidebar.write("### Opciones para Procesamiento de DICOM")

        # Subselección entre "Visualización de DICOM" y "Convertir a PNG"
        subseccion = st.sidebar.radio(
            "Selecciona la subsección",
            ["Visualización de DICOM", "Convertir a PNG"]
        )
        opciones['subseccion'] = subseccion

        if subseccion == "Visualización de DICOM":
            uploaded_files = st.sidebar.file_uploader(
                "Cargar archivos DICOM",
                type=["dcm", "dicom"],
                accept_multiple_files=True
            )
            opciones['uploaded_files'] = uploaded_files
            opciones['mostrar_metadatos'] = st.sidebar.checkbox("Mostrar Metadatos", value=False)
            opciones['aplicar_voilut'] = st.sidebar.checkbox("Aplicar VOI LUT", value=False)
            opciones['invertir_interpretacion'] = st.sidebar.checkbox("Invertir Interpretación Fotométrica",
                                                                      value=False)
            opciones['aplicar_transformaciones'] = st.sidebar.checkbox("Aplicar Transformaciones", value=False)

            if opciones['aplicar_transformaciones']:
                st.sidebar.write("### Selecciona las Transformaciones a Aplicar")

                transformaciones = [
                    ('voltear_horizontal', "Volteo Horizontal"),
                    ('voltear_vertical', "Volteo Vertical"),
                    ('brillo_contraste', "Ajuste de Brillo y Contraste"),
                    ('ruido_gaussiano', "Añadir Ruido Gaussiano"),
                    ('recorte_redimension', "Recorte Aleatorio y Redimensionado"),
                    ('desenfoque', "Aplicar Desenfoque")
                ]

                opciones['transformaciones_seleccionadas'] = {}

                for key, label in transformaciones:
                    opciones['transformaciones_seleccionadas'][key] = st.sidebar.checkbox(label=label, value=False,
                                                                                          key=key)

        elif subseccion == "Convertir a PNG":
            mostrar_convertir_png(opciones)

    elif tipo_carga == "Clasificación mediante Deep Learning":
        st.sidebar.write("### Opciones para Clasificación mediante Deep Learning")

        # Subir una imagen (DICOM, PNG, JPG)
        uploaded_image = st.sidebar.file_uploader(
            "Cargar imagen (DICOM, PNG, JPG)",
            type=["dcm", "dicom", "png", "jpg", "jpeg"],
            accept_multiple_files=False
        )

        if uploaded_image is not None:
            # Procesar la imagen
            image, tipo_archivo = procesar_archivo(uploaded_image)

            if image:
                st.image(image, caption='Imagen procesada (224x224)', use_column_width=True)

                # Definir las rutas y file_urls para los tres modelos
                modelos_info = {
                    'primario': {
                        'model_folder': 'ViT-large-patch16-224_B',
                        'file_url': 'https://usmcl-my.sharepoint.com/:u:/g/personal/julio_maturana_usm_cl/EVMIWphh_1ZIrDG6VeKXZX0BIT3vlDBoensMcRx-YTve3w?e=WTaQdV'
                        # Reemplaza con el URL de descarga directo de tu modelo primario
                    },
                    'secondary_masas': {
                        'model_folder': 'VT_V8',
                        'file_url': 'https://usmcl-my.sharepoint.com/:u:/g/personal/julio_maturana_usm_cl/INSERT_SECONDARY_MASAS_URL_HERE?e=XXXXXX'
                        # Reemplaza con el URL de descarga directo de tu modelo secundario para masas
                    },
                    'secondary_calcifi': {
                        'model_folder': 'Cal_ViT-large-patch16-224_A.ipynb',
                        'file_url': 'https://usmcl-my.sharepoint.com/:u:/g/personal/julio_maturana_usm_cl/EbGZhS3H-XFHvOFJKsTmz4sBX2g7OqtrtnaSzlap3b0h5Q?e=iuNG1y'
                        # Reemplaza con el URL de descarga directo de tu modelo CALCI
                    }
                }

                # Descargar y cargar los modelos
                classifiers = {}
                prediction_mappings = {}

                for key, info in modelos_info.items():
                    model_path = descargar_modelo(
                        model_dir=os.path.join('src', 'data', 'modelos'),
                        model_folder=info['model_folder'],
                        file_url=info['file_url']
                    )
                    if model_path:
                        if key == 'primario':
                            classifiers['primario'] = cargar_modelo_primary(model_path)
                            prediction_mappings['primario'] = {
                                'LABEL_0': 'masas',
                                'LABEL_1': 'calcificaciones',
                                'LABEL_2': 'no_encontrado'
                            }
                        elif key == 'secondary_masas':
                            classifiers['secondary_masas'] = cargar_modelo_secondary_masas(model_path)
                            prediction_mappings['secondary_masas'] = {
                                'LABEL_0': 'benigna',
                                'LABEL_1': 'maligna'
                            }
                        elif key == 'secondary_calcifi':
                            classifiers['secondary_calcifi'] = cargar_modelo_secondary_calcifi(model_path)
                            prediction_mappings['secondary_calcifi'] = {
                                'LABEL_0': 'benigna',
                                'LABEL_1': 'maligna',
                                'LABEL_2': 'sospechosa'
                            }

                # Verificar que el modelo primario se ha cargado correctamente
                if 'primario' in classifiers and classifiers['primario']:
                    # Realizar la inferencia primaria
                    mapped_result_primary = clasificar_imagen(image, classifiers['primario'],
                                                              prediction_mappings['primario'])

                    # Mostrar los resultados de la clasificación primaria
                    mostrar_resultados_primary(mapped_result_primary)

                    # Lógica para clasificación secundaria según la clasificación primaria
                    if mapped_result_primary:
                        # Determinar la etiqueta con mayor puntuación
                        primary_label = max(mapped_result_primary, key=mapped_result_primary.get)
                        if primary_label == 'masas':
                            if 'secondary_masas' in classifiers and classifiers['secondary_masas']:
                                # Realizar la inferencia secundaria para masas
                                mapped_result_secondary_masas = clasificar_imagen(image, classifiers['secondary_masas'],
                                                                                  prediction_mappings[
                                                                                      'secondary_masas'])
                                # Mostrar los resultados de la clasificación secundaria para masas
                                mostrar_resultados_secondary_masas(mapped_result_secondary_masas)
                            else:
                                st.error("No se pudo cargar el modelo secundario para la clasificación de masas.")

                        elif primary_label == 'calcificaciones':
                            if 'secondary_calcifi' in classifiers and classifiers['secondary_calcifi']:
                                # Realizar la inferencia secundaria para calcificaciones
                                mapped_result_secondary_calcifi = clasificar_imagen(image,
                                                                                    classifiers['secondary_calcifi'],
                                                                                    prediction_mappings[
                                                                                        'secondary_calcifi'])
                                # Mostrar los resultados de la clasificación secundaria para calcificaciones
                                mostrar_resultados_secondary_calcifi(mapped_result_secondary_calcifi)
                            else:
                                st.error(
                                    "No se pudo cargar el modelo secundario para la clasificación de calcificaciones.")
                        elif primary_label == 'no_encontrado':
                            st.write("### La imagen no contiene masas ni calcificaciones detectadas.")
                else:
                    st.error("No se pudo cargar el modelo primario para la clasificación.")
        else:
            st.info("Por favor, carga una imagen DICOM, PNG o JPG para realizar la clasificación.")

    if __name__ == "__main__":
        main()
