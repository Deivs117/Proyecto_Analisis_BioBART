# model_loader.py — Carga del tokenizador y modelo BioBART con caché de Streamlit
# ─────────────────────────────────────────────────────────────────────────────
# El decorador @st.cache_resource garantiza que el modelo se descarga y carga
# UNA SOLA VEZ por sesión de Streamlit, evitando re-cargas costosas en cada
# interacción del usuario.

import torch
import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration

from config import MODEL_NAME


def detectar_dispositivo() -> torch.device:
    """Detecta si hay GPU disponible; si no, usa CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@st.cache_resource(show_spinner="⏳ Cargando modelo BioBART… (solo ocurre la primera vez)")
def cargar_modelo():
    """
    Descarga (si no está en caché local) y carga el tokenizador y el modelo
    BioBART desde HuggingFace Hub.

    Returns
    -------
    tuple[BartTokenizer, BartForConditionalGeneration, torch.device]
        tokenizador, modelo en modo evaluación y dispositivo activo.
    """
    dispositivo = detectar_dispositivo()

    tokenizador = BartTokenizer.from_pretrained(MODEL_NAME)

    modelo = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    modelo.to(dispositivo)
    modelo.eval()  # desactiva dropout y batchnorm en modo inferencia

    return tokenizador, modelo, dispositivo
