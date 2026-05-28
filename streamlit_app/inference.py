# inference.py — Funciones de inferencia para resumen y QA con BioBART
# ─────────────────────────────────────────────────────────────────────────────

import torch
from transformers import BartTokenizer, BartForConditionalGeneration

from config import MAX_INPUT_TOKENS, SUMMARIZATION_PARAMS, QA_PARAMS


def _tokenizar(
    texto: str,
    tokenizador: BartTokenizer,
    dispositivo: torch.device,
) -> dict:
    """
    Tokeniza un texto de entrada aplicando truncamiento al límite del modelo.

    Parameters
    ----------
    texto : str
        Texto de entrada (reporte o prompt de QA).
    tokenizador : BartTokenizer
    dispositivo : torch.device

    Returns
    -------
    dict
        Diccionario con 'input_ids' y 'attention_mask' como tensores en el
        dispositivo correcto.
    """
    return tokenizador(
        texto,
        return_tensors="pt",
        max_length=MAX_INPUT_TOKENS,
        truncation=True,
    ).to(dispositivo)


def generar_resumen(
    reporte: str,
    tokenizador: BartTokenizer,
    modelo: BartForConditionalGeneration,
    dispositivo: torch.device,
    params: dict | None = None,
) -> tuple[str, int]:
    """
    Genera un resumen de un reporte radiológico/clínico usando Beam Search.

    Parameters
    ----------
    reporte : str
        Texto del reporte a resumir.
    tokenizador : BartTokenizer
    modelo : BartForConditionalGeneration
    dispositivo : torch.device
    params : dict, opcional
        Parámetros de generación; si es None usa los valores de config.py.

    Returns
    -------
    tuple[str, int]
        (texto del resumen generado, número de tokens de entrada)
    """
    if params is None:
        params = SUMMARIZATION_PARAMS

    inputs = _tokenizar(reporte, tokenizador, dispositivo)
    n_tokens_entrada = inputs["input_ids"].shape[1]

    with torch.no_grad():
        salida_ids = modelo.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **params,
        )

    resumen = tokenizador.decode(salida_ids[0], skip_special_tokens=True)
    return resumen, n_tokens_entrada


def generar_respuesta_qa(
    pregunta: str,
    contexto: str,
    tokenizador: BartTokenizer,
    modelo: BartForConditionalGeneration,
    dispositivo: torch.device,
    params: dict | None = None,
) -> tuple[str, int]:
    """
    Genera una respuesta a una pregunta médica dado un contexto clínico,
    usando Nucleus Sampling (do_sample=True) para mayor naturalidad.

    El prompt sigue la convención: "question: <pregunta> context: <contexto>"

    Parameters
    ----------
    pregunta : str
        Pregunta médica.
    contexto : str
        Contexto clínico/radiológico de referencia.
    tokenizador : BartTokenizer
    modelo : BartForConditionalGeneration
    dispositivo : torch.device
    params : dict, opcional
        Parámetros de generación; si es None usa los valores de config.py.

    Returns
    -------
    tuple[str, int]
        (texto de la respuesta generada, número de tokens del prompt)
    """
    if params is None:
        params = QA_PARAMS

    prompt = f"question: {pregunta.strip()} context: {contexto.strip()}"
    inputs = _tokenizar(prompt, tokenizador, dispositivo)
    n_tokens_entrada = inputs["input_ids"].shape[1]

    with torch.no_grad():
        salida_ids = modelo.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **params,
        )

    respuesta = tokenizador.decode(salida_ids[0], skip_special_tokens=True)
    return respuesta, n_tokens_entrada
