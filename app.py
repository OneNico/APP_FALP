# app.pyy

import streamlit as st
from src.ui.visualizacion import mostrar_visualizacion
from src.ui.convertir_png import mostrar_convertir_png  # Importar la nueva función
from src.ui.clasificacion_deep_learning import (
    procesar_archivo,
    clasificar_imagen,
    mostrar_resultados
)
from src.utilidades.importar_modelo import descargar_modelo, cargar_modelo
from PIL import Image
import os
import logging

# Configuración del logger
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def main():
    # Cargar el archivo CSS externo
    def cargar_css():
        try:
            with open('styles/style.css') as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        except FileNotFoundError:
            st.error("No se encontró el archivo 'styles/style.css'.")

    cargar_css()

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

    st.sidebar.header("Opciones de Procesamiento")
    tipo_carga = st.sidebar.radio(
        "Selecciona el tipo de carga",
        ["Procesamiento de DICOM", "Clasificación mediante Deep Learning"]
    )

    opciones = {'tipo_carga': tipo_carga}

    if tipo_carga == "Procesamiento de DICOM":
        st.sidebar.write("### Opciones para Procesamiento de DICOM")

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
            opciones['invertir_interpretacion'] = st.sidebar.checkbox("Invertir Interpretación Fotométrica", value=False)
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
                    opciones['transformaciones_seleccionadas'][key] = st.sidebar.checkbox(label=label, value=False, key=key)

        elif subseccion == "Convertir a PNG":
            mostrar_convertir_png(opciones)

    elif tipo_carga == "Clasificación mediante Deep Learning":
        st.sidebar.write("### Opciones para Clasificación mediante Deep Learning")

        uploaded_image = st.sidebar.file_uploader(
            "Cargar imagen (DICOM, PNG, JPG)",
            type=["dcm", "dicom", "png", "jpg", "jpeg"],
            accept_multiple_files=False
        )

        if uploaded_image is not None:
            image, tipo_archivo = procesar_archivo(uploaded_image)

            if image:
                st.image(image, caption='Imagen procesada (224x224)', use_column_width=True)

                with st.spinner("Cargando el modelo (solo lo realizará la primera vez)..."):
                    # Inicializar el pipeline de clasificación de imágenes
                    model_dir = os.path.join('src', 'data', 'modelos')
                    model_folder = 'VT_V8'
                    file_id_zip = "1S4oBDDV0KGdJQVllj6kmz4pekubVVg-J"

                    # Descargar el modelo si no existe
                    model_path = descargar_modelo(model_dir, model_folder, file_id_zip)

                    if model_path:
                        # Cargar el modelo desde el nuevo módulo
                        classifier = cargar_modelo(model_path)

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

    if tipo_carga == "Procesamiento de DICOM":
        if opciones.get('subseccion') == "Visualización de DICOM":
            mostrar_visualizacion(opciones)
        elif opciones.get('subseccion') == "Convertir a PNG":
            pass
    else:
        pass

if __name__ == "__main__":
    main()
