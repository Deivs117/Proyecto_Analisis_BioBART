# config.py — Constantes y parámetros de inferencia por defecto
# ─────────────────────────────────────────────────────────────────────────────

# ── Modelo ────────────────────────────────────────────────────────────────────
# Checkpoint principal de BioBART usado en el notebook de inferencia.
# GanjinZero/biobart-base  (~560 MB) — recomendado para CPU
# GanjinZero/biobart-large (~1.6 GB) — recomendado si se dispone de GPU
MODEL_NAME = "GanjinZero/biobart-base"

# Longitud máxima de tokens de entrada que acepta el modelo (límite arquitectural)
MAX_INPUT_TOKENS = 512

# ── Parámetros de resumen (Beam Search) ──────────────────────────────────────
SUMMARIZATION_PARAMS = {
    "max_new_tokens": 150,
    "num_beams": 4,
    "min_length": 30,
    "no_repeat_ngram_size": 3,
    "length_penalty": 2.0,
    "early_stopping": True,
}

# ── Parámetros de QA médico (Nucleus Sampling) ───────────────────────────────
QA_PARAMS = {
    "max_new_tokens": 200,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.92,
    "top_k": 50,
    "no_repeat_ngram_size": 3,
}

# ── Ejemplos precargados ──────────────────────────────────────────────────────
EJEMPLO_REPORTE = (
    "Patient is a 54-year-old female presenting with oppressive chest pain "
    "radiating to the left arm for 2 hours. ECG shows ST-segment elevation "
    "in V1-V4. Troponin I: 2.8 ng/mL. Blood pressure: 95/60 mmHg. "
    "Medical history: hypertension, type 2 diabetes mellitus. "
    "Initial management with aspirin 300 mg, heparin IV, and urgent "
    "catheterization lab activation. Diagnosis: STEMI."
)

EJEMPLO_PREGUNTA = "What is the most likely diagnosis based on the clinical findings?"

EJEMPLO_CONTEXTO = (
    "A 67-year-old male with a history of smoking and hypertension presents "
    "with acute onset shortness of breath and hemoptysis. CT pulmonary "
    "angiography reveals a filling defect in the right pulmonary artery. "
    "D-dimer is markedly elevated at 4500 ng/mL. Heart rate 110 bpm, "
    "oxygen saturation 88% on room air."
)

# ── Aviso / Disclaimer ────────────────────────────────────────────────────────
DISCLAIMER = (
    "⚠️ **AVISO IMPORTANTE:** Esta es una **demo académica** de inferencia "
    "con BioBART. Las salidas generadas **no deben usarse para diagnóstico "
    "clínico real**. El modelo fue pre-entrenado en inglés sobre abstracts "
    "de PubMed; los textos en español pueden producir resultados degradados. "
    "Reportes superiores a 512 tokens serán truncados automáticamente."
)
