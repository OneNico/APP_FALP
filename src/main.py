# src/main.py

import streamlit as st
from src.ui.sidebar import mostrar_sidebar
from src.ui.visualizacion import mostrar_visualizacion
from src.config.logging_config import setup_logging
import os


def main():
    # Configurar el sistema de logging
    setup_logging()

    # Configurar la página de Streamlit
    st.set_page_config(
        page_title="Módulo de Procesamiento de Imágenes",
        layout="wide"
    )

    # Aplicar estilos CSS personalizados
    aplicar_estilos_css()

    # Asegurar que la carpeta data/temp existe
    ruta_temp = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'temp')
    os.makedirs(ruta_temp, exist_ok=True)

    # Título principal
    st.title("Módulo de Procesamiento de Imágenes")

    # Mostrar opciones en la barra lateral y obtener selecciones
    opciones = mostrar_sidebar()

    # Mostrar la interfaz de visualización
    mostrar_visualizacion(opciones)


def aplicar_estilos_css():
    """
    Lee y aplica estilos CSS personalizados desde el archivo 'styles/style.css'.
    """
    css_file = os.path.join(os.path.dirname(__file__), '..', 'styles', 'style.css')
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.warning("Archivo de estilos CSS no encontrado.")
