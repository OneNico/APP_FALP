import streamlit as st
from src.ui.visualizacion import mostrar_visualizacion
from src.ui.convertir_png import mostrar_convertir_png
from src.ui.clasificacion_deep_learning import (
    procesar_archivo,
    clasificar_imagen,
    mostrar_resultados
)
import os
import torch
import gdown  # Para descargar el modelo desde Google Drive
from safetensors.torch import load_file  # Para cargar modelos .safetensors


# Función para descargar y cargar el modelo desde Google Drive
def cargar_modelo_safetensors(model_path, google_drive_url=None):
    # Si el modelo no está en la ruta local, descargarlo desde Google Drive
    if not os.path.exists(model_path) and google_drive_url is not None:
        st.info("Descargando el modelo desde Google Drive...")
        gdown.download(google_drive_url, model_path, quiet=False)

    # Cargar el modelo desde la ruta local (formato .safetensors)
    if os.path.exists(model_path):
        st.success(f"Modelo cargado desde {model_path}")
        model = load_file(model_path)  # Cargar el modelo en formato .safetensors
        return model
    else:
        st.error("El modelo no se pudo encontrar o descargar.")
        return None


def main():
    # Inyectar CSS personalizado
    css = """
    <style>
    /* Tu código CSS personalizado aquí */
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    # Título personalizado
    title_html = """
    <div class="outer">
        <div class="dot"></div>
        <div class="card">
            <div class="ray"></div>
            <div class="text">Clasificación de mamografías</div>
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

    if tipo_carga == "Clasificación mediante Deep Learning":
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

                # Ruta temporal donde se guardará el modelo descargado
                model_path = "model.safetensors"

                # Enlace de descarga directa desde Google Drive para el archivo .safetensors
                google_drive_url = "https://drive.google.com/uc?export=download&id=1zS3NKksvdnDxr2JHTrABClLmi4q324GI"

                # Cargar el modelo (.safetensors desde Google Drive si no está en la ruta local)
                classifier = cargar_modelo_safetensors(model_path, google_drive_url)

                if classifier:
                    prediction_mapping = {
                        'LABEL_0': 'benigna',
                        'LABEL_1': 'maligna'
                    }

                    # Realizar la inferencia (si el modelo es compatible con esta estructura)
                    mapped_result = clasificar_imagen(image, classifier, prediction_mapping)

                    # Mostrar los resultados
                    mostrar_resultados(mapped_result)
        else:
            st.info("Por favor, carga una imagen DICOM, PNG o JPG para realizar la clasificación.")

if __name__ == "__main__":
    main()
