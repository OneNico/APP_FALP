# app.py

import streamlit as st
from src.ui.visualizacion import mostrar_visualizacion
from src.ui.convertir_png import mostrar_convertir_png  # Importar la nueva función
from src.ui.clasificacion_deep_learning import (
    cargar_modelo,
    cargar_modelo_pytorch,  # Asegúrate de tener esta función en tu módulo
    procesar_archivo,
    clasificar_imagen,
    mostrar_resultados
)
from PIL import Image
import os
import torch
import gdown
import zipfile
import logging

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
    if os.path.exists(model_path):
        st.info("El modelo ya está presente localmente.")
        return model_path

    st.info("Descargando el modelo desde Google Drive. Esto puede tardar unos minutos...")
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    zip_path = os.path.join(model_dir, f"{model_folder}.zip")

    try:
        gdown.download(url, zip_path, quiet=False)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        os.remove(zip_path)
        st.success("Modelo descargado y extraído correctamente.")
        return model_path
    except Exception as e:
        logger.error(f"Error al descargar o extraer el modelo: {e}")
        st.error(f"Error al descargar o extraer el modelo: {e}")
        return None

def listar_archivos(model_path):
    """
    Lista todos los archivos y carpetas dentro del directorio del modelo.
    
    :param model_path: Ruta al directorio del modelo.
    """
    st.write("### Archivos en el directorio del modelo:")
    for root, dirs, files in os.walk(model_path):
        level = root.replace(model_path, '').count(os.sep)
        indent = ' ' * 4 * level
        st.write(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            st.write(f"{subindent}{f}")

def main():
    # Inyectar CSS personalizado para estilos profesionales y el nuevo diseño del título
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@700&display=swap');

    /* Estilos Generales */
    .stImage > img {
        border: 2px solid #00BFFF; /* Celeste */
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
    }
    .stExpander > div {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 5px;
        font-family: 'Roboto', sans-serif;
    }
    .main .block-container{
        padding-top: 2rem;
        padding-right: 3rem;
        padding-left: 3rem;
        padding-bottom: 2rem;
    }

    /* From Uiverse.io by Spacious74 */ 
    .outer {
      width: 600px; /* Aumentado de 300px a 600px */
      height: 200px; /* Reducido de 400px a 200px */
      border-radius: 10px;
      padding: 1px;
      background: radial-gradient(circle 230px at 0% 0%, #ffffff, #0c0d0d);
      position: relative;
      margin: 0 auto 50px auto; /* Centrado horizontalmente con margen inferior */
      overflow: hidden; /* Evitar que elementos animados salgan del contenedor */
    }

    .dot {
      width: 5px; /* Mantenido en 5px */
      aspect-ratio: 1;
      position: absolute;
      background-color: #fff;
      box-shadow: 0 0 10px #ffffff;
      border-radius: 100px;
      z-index: 2;
      right: 10%;
      top: 10%;
      animation: moveDot 6s linear infinite;
    }

    @keyframes moveDot {
      0%,
      100% {
        top: 10%;
        right: 10%;
      }
      25% {
        top: 10%;
        right: calc(100% - 35px);
      }
      50% {
        top: calc(100% - 25px); /* Ajustado para el nuevo height */
        right: calc(100% - 35px);
      }
      75% {
        top: calc(100% - 25px); /* Ajustado para el nuevo height */
        right: 10%;
      }
    }

    .card {
      z-index: 1;
      width: 100%;
      height: 100%;
      border-radius: 9px;
      border: solid 1px #202222;
      background-size: 20px 20px;
      background: radial-gradient(circle 280px at 0% 0%, #444444, #0c0d0d);
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
      flex-direction: column;
      color: #fff;
    }
    .ray {
      width: 150px; /* Reducido de 300px a 150px */
      height: 30px;  /* Reducido de 60px a 30px */
      border-radius: 100px;
      position: absolute;
      background-color: #c7c7c7;
      opacity: 0.4;
      box-shadow: 0 0 50px #fff;
      filter: blur(10px);
      transform-origin: 10%;
      top: 0%;
      left: 0;
      transform: rotate(40deg);
      animation: moveRay 6s linear infinite;
    }

    @keyframes moveRay {
        0% { transform: rotate(40deg); }
        50% { transform: rotate(-40deg); }
        100% { transform: rotate(40deg); }
    }

    .card .text {
      font-weight: bolder;
      font-size: 2.5rem; /* Reducido de 3rem a 2.5rem para mejor ajuste */
      background: linear-gradient(45deg, #000000 4%, #fff, #000);
      background-clip: text;
      color: transparent;
      margin-bottom: 5px; /* Reducido de 10px a 5px para mejor ajuste */
      text-align: center; /* Centrado del texto */
    }

    .line {
      width: 100%;
      height: 1px;
      position: absolute;
      background-color: #2c2c2c;
    }
    .topl {
      top: 10%;
      background: linear-gradient(90deg, #888888 30%, #1d1f1f 70%);
    }
    .bottoml {
      bottom: 10%;
      background: linear-gradient(90deg, #888888 30%, #1d1f1f 70%);
    }
    .leftl {
      left: 10%;
      width: 1px;
      height: 100%;
      background: linear-gradient(180deg, #747474 30%, #222424 70%);
    }
    .rightl {
      right: 10%;
      width: 1px;
      height: 100%;
      background: linear-gradient(180deg, #747474 30%, #222424 70%);
    }

    /* Estilos para Toggle Switches */
    .toggle-container {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }

    .toggle-container input[type="checkbox"] {
        display: none;
    }

    .toggle-container label {
        position: relative;
        display: inline-block;
        width: 50px;
        height: 24px;
        background-color: #ccc;
        border-radius: 24px;
        cursor: pointer;
        transition: background-color 0.2s;
    }

    .toggle-container label::after {
        content: "";
        position: absolute;
        width: 20px;
        height: 20px;
        left: 2px;
        top: 2px;
        background-color: white;
        border-radius: 50%;
        transition: transform 0.2s;
    }

    .toggle-container input[type="checkbox"]:checked + label {
        background-color: #00BFFF;
    }

    .toggle-container input[type="checkbox"]:checked + label::after {
        transform: translateX(26px);
    }

    .toggle-label {
        margin-left: 10px;
        font-family: 'Roboto', sans-serif;
        font-size: 14px;
        color: #333;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    # HTML para el título dentro de la estructura proporcionada
    title_html = """
    <div class="outer">
        <div class="dot"></div>
        <div class="card">
            <div class="ray"></div>
            <div class="text">Clasificación de mamografias</div>
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
            # Opciones para Visualización de DICOM
            uploaded_files = st.sidebar.file_uploader(
                "Cargar archivos DICOM",
                type=["dcm", "dicom"],
                accept_multiple_files=True
            )
            opciones['uploaded_files'] = uploaded_files

            # Opciones adicionales
            opciones['mostrar_metadatos'] = st.sidebar.checkbox("Mostrar Metadatos", value=False)
            opciones['aplicar_voilut'] = st.sidebar.checkbox("Aplicar VOI LUT", value=False)  # Cambiado a False por defecto
            opciones['invertir_interpretacion'] = st.sidebar.checkbox("Invertir Interpretación Fotométrica", value=False)
            opciones['aplicar_transformaciones'] = st.sidebar.checkbox("Aplicar Transformaciones", value=False)

            # Si se selecciona aplicar transformaciones, mostrar las 6 opciones restantes con checkboxes
            if opciones['aplicar_transformaciones']:
                st.sidebar.write("### Selecciona las Transformaciones a Aplicar")

                # Crear una lista de transformaciones
                transformaciones = [
                    ('voltear_horizontal', "Volteo Horizontal"),
                    ('voltear_vertical', "Volteo Vertical"),
                    ('brillo_contraste', "Ajuste de Brillo y Contraste"),
                    ('ruido_gaussiano', "Añadir Ruido Gaussiano"),
                    ('recorte_redimension', "Recorte Aleatorio y Redimensionado"),
                    ('desenfoque', "Aplicar Desenfoque")
                ]

                # Diccionario para almacenar las selecciones
                opciones['transformaciones_seleccionadas'] = {}

                for key, label in transformaciones:
                    # Crear un checkbox para cada transformación
                    opciones['transformaciones_seleccionadas'][key] = st.sidebar.checkbox(label=label, value=False, key=key)

        elif subseccion == "Convertir a PNG":
            # Opciones para Convertir a PNG
            mostrar_convertir_png(opciones)  # Mostrar la nueva sección

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
                # Mostrar la imagen procesada
                st.image(image, caption='Imagen procesada (224x224)', use_column_width=True)

                # Inicializar el pipeline de clasificación de imágenes
                model_dir = os.path.join('src', 'data', 'modelos')
                model_folder = 'VT_V8'  # Cambiado a 'VT_V8'
                model_path = os.path.join(model_dir, model_folder)

                # ID del archivo ZIP en Google Drive (reemplaza con tu ID real del ZIP)
                file_id_zip = "1S4oBDDV0KGdJQVllj6kmz4pekubVVg-J"  # Reemplaza con el ID real del ZIP

                # Descargar el modelo si no existe
                model_path = descargar_modelo(model_dir, model_folder, file_id_zip)

                if model_path:
                    # Listar archivos para depuración
                    listar_archivos(model_path)

                    # Cargar el modelo
                    classifier = cargar_modelo(model_path)

                    if classifier is None:
                        # Si falla, intentar cargarlo manualmente como un modelo de PyTorch
                        classifier = cargar_modelo_pytorch(model_path)

                    if classifier:
                        # Definir mapeo de etiquetas
                        prediction_mapping = {
                            'LABEL_0': 'benigna',
                            'LABEL_1': 'maligna'
                        }

                        # Realizar la inferencia
                        mapped_result = clasificar_imagen(image, classifier, prediction_mapping)

                        # Mostrar los resultados
                        mostrar_resultados(mapped_result)
        else:
            st.info("Por favor, carga una imagen DICOM, PNG o JPG para realizar la clasificación.")

    # Mostrar la visualización de imágenes o la sección de convertir a PNG
    if tipo_carga == "Procesamiento de DICOM":
        if opciones.get('subseccion') == "Visualización de DICOM":
            mostrar_visualizacion(opciones)
        elif opciones.get('subseccion') == "Convertir a PNG":
            pass  # La lógica de 'Convertir a PNG' se maneja dentro de 'mostrar_convertir_png'
    else:
        pass  # La lógica para "Clasificación mediante Deep Learning" ya está manejada arriba

if __name__ == "__main__":
    main()
