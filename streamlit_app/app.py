# app.py — Interfaz Streamlit principal para demo de BioBART
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st

from config import (
    DISCLAIMER,
    EJEMPLO_CONTEXTO,
    EJEMPLO_PREGUNTA,
    EJEMPLO_REPORTE,
    GREEDY_PARAMS,
    MODEL_NAME,
    QA_PARAMS,
    SUMMARIZATION_PARAMS,
)
from inference import generar_respuesta_qa, generar_resumen
from model_loader import cargar_modelo

# ─────────────────────────────────────────────────────────────────────────────
# Configuración de la página
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BioBART — Demo de Inferencia",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Cabecera principal
# ─────────────────────────────────────────────────────────────────────────────
st.title("🏥 BioBART — Demo de Inferencia Biomédica")
st.markdown(
    "**Demo académica** de generación de texto clínico con [BioBART]"
    "(https://arxiv.org/abs/2204.03905) · Materia: Procesamiento de Datos Secuenciales"
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Carga del modelo (se ejecuta UNA SOLA VEZ gracias a @st.cache_resource)
# ─────────────────────────────────────────────────────────────────────────────
tokenizador, modelo, dispositivo = cargar_modelo()

# ─────────────────────────────────────────────────────────────────────────────
# Barra lateral — información del sistema y disclaimer
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Información del sistema")
    st.markdown(f"**Modelo:** `{MODEL_NAME}`")
    st.markdown(f"**Dispositivo:** `{dispositivo}`")
    st.markdown("**Ventana máxima:** 512 tokens de entrada")
    st.markdown("---")
    st.markdown("### 📋 Modos disponibles")
    st.markdown(
        "1. **Resumen radiológico** — Beam Search\n"
        "2. **QA médico** — Beam Search\n"
        "3. **Demo comparativa** — Greedy Search vs. Beam Search"
    )
    st.markdown("---")
    st.warning(DISCLAIMER)

# ─────────────────────────────────────────────────────────────────────────────
# Selector de modo mediante pestañas
# ─────────────────────────────────────────────────────────────────────────────
tab_resumen, tab_qa, tab_comparativa = st.tabs(
    ["📄 Resumen Radiológico", "❓ Pregunta / Respuesta Médica", "⚖️ Demo Comparativa"]
)


# ═════════════════════════════════════════════════════════════════════════════
# PESTAÑA 1 — Resumen radiológico
# ═════════════════════════════════════════════════════════════════════════════
with tab_resumen:
    st.subheader("📄 Resumen automático de reporte radiológico / clínico")
    st.markdown(
        "Pega un reporte clínico o radiológico en inglés y BioBART generará "
        "un resumen abstractivo usando **Beam Search** (`num_beams=4`)."
    )

    # ── Controles de ejemplo / limpiar ────────────────────────────────────
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("💡 Cargar ejemplo", key="ejemplo_resumen"):
            st.session_state["input_resumen"] = EJEMPLO_REPORTE
    with col2:
        if st.button("🗑️ Limpiar", key="limpiar_resumen"):
            st.session_state["input_resumen"] = ""

    # ── Área de texto ─────────────────────────────────────────────────────
    texto_resumen = st.text_area(
        "Reporte clínico / radiológico",
        height=220,
        placeholder=(
            "Pega aquí el reporte en inglés. "
            "Ejemplo: 'Patient is a 65-year-old male with chest pain...'"
        ),
        key="input_resumen",
    )

    # ── Parámetros avanzados (desplegable) ────────────────────────────────
    with st.expander("⚙️ Parámetros de generación (Beam Search)"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            max_new_tokens_r = st.slider(
                "max_new_tokens", 50, 300, SUMMARIZATION_PARAMS["max_new_tokens"], 10
            )
            num_beams = st.slider("num_beams", 1, 8, SUMMARIZATION_PARAMS["num_beams"])
        with col_b:
            min_length = st.slider(
                "min_length", 10, 100, SUMMARIZATION_PARAMS["min_length"], 5
            )
            no_repeat_ngram = st.slider(
                "no_repeat_ngram_size", 1, 5, SUMMARIZATION_PARAMS["no_repeat_ngram_size"]
            )
        with col_c:
            length_penalty = st.slider(
                "length_penalty", 0.5, 4.0, SUMMARIZATION_PARAMS["length_penalty"], 0.1
            )
            early_stopping = st.checkbox(
                "early_stopping", value=SUMMARIZATION_PARAMS["early_stopping"]
            )

    params_resumen = {
        "max_new_tokens": max_new_tokens_r,
        "num_beams": num_beams,
        "min_length": min_length,
        "no_repeat_ngram_size": no_repeat_ngram,
        "length_penalty": length_penalty,
        "early_stopping": early_stopping,
    }

    # ── Botón de inferencia ───────────────────────────────────────────────
    if st.button("🔬 Generar resumen", type="primary", key="btn_resumen"):
        texto_limpio = texto_resumen.strip()
        if not texto_limpio:
            st.warning("Por favor ingresa un reporte antes de generar el resumen.")
        else:
            with st.spinner("Generando resumen… esto puede tardar unos segundos en CPU."):
                try:
                    resumen, n_tokens = generar_resumen(
                        texto_limpio, tokenizador, modelo, dispositivo, params_resumen
                    )
                    st.success("✅ Resumen generado correctamente.")
                    st.markdown("### 📝 Resumen generado")
                    st.info(resumen)
                    st.caption(f"📊 Tokens de entrada utilizados: **{n_tokens}** / 512")
                    if n_tokens >= 512:
                        st.warning(
                            "⚠️ El reporte fue truncado a 512 tokens. "
                            "Es posible que se haya perdido información del final del texto."
                        )
                except Exception as exc:
                    st.error(f"❌ Error durante la inferencia: {exc}")


# ═════════════════════════════════════════════════════════════════════════════
# PESTAÑA 2 — QA médico
# ═════════════════════════════════════════════════════════════════════════════
with tab_qa:
    st.subheader("❓ Pregunta / Respuesta médica radiológica")
    st.markdown(
        "Ingresa una **pregunta clínica** y el **contexto** del caso. "
        "BioBART generará una respuesta usando **Beam Search** "
        "(`num_beams=4`) para mayor coherencia y fidelidad factual."
    )

    # ── Controles de ejemplo / limpiar ────────────────────────────────────
    col3, col4 = st.columns([1, 1])
    with col3:
        if st.button("💡 Cargar ejemplo", key="ejemplo_qa"):
            st.session_state["input_pregunta"] = EJEMPLO_PREGUNTA
            st.session_state["input_contexto"] = EJEMPLO_CONTEXTO
    with col4:
        if st.button("🗑️ Limpiar", key="limpiar_qa"):
            st.session_state["input_pregunta"] = ""
            st.session_state["input_contexto"] = ""

    # ── Campos de entrada ─────────────────────────────────────────────────
    pregunta_qa = st.text_input(
        "Pregunta clínica (en inglés)",
        placeholder="What is the most likely diagnosis based on the findings?",
        key="input_pregunta",
    )

    contexto_qa = st.text_area(
        "Contexto clínico / radiológico (en inglés)",
        height=200,
        placeholder=(
            "Pega aquí la nota clínica o hallazgos radiológicos relevantes para responder "
            "la pregunta. Ejemplo: 'A 67-year-old male presents with shortness of breath...'"
        ),
        key="input_contexto",
    )

    # ── Parámetros avanzados (desplegable) ────────────────────────────────
    with st.expander("⚙️ Parámetros de generación (Beam Search)"):
        col_d, col_e = st.columns(2)
        with col_d:
            max_new_tokens_q = st.slider(
                "max_new_tokens", 50, 400, QA_PARAMS["max_new_tokens"], 10
            )
            num_beams_q = st.slider("num_beams", 1, 8, QA_PARAMS["num_beams"])
        with col_e:
            min_length_q = st.slider(
                "min_length", 0, 80, QA_PARAMS["min_length"], 5
            )
            no_repeat_ngram_q = st.slider(
                "no_repeat_ngram_size", 1, 5, QA_PARAMS["no_repeat_ngram_size"]
            )

    params_qa = {
        "max_new_tokens": max_new_tokens_q,
        "num_beams": num_beams_q,
        "min_length": min_length_q,
        "no_repeat_ngram_size": no_repeat_ngram_q,
        "early_stopping": True,
    }

    # ── Botón de inferencia ───────────────────────────────────────────────
    if st.button("💬 Generar respuesta", type="primary", key="btn_qa"):
        pregunta_limpia = pregunta_qa.strip()
        contexto_limpio = contexto_qa.strip()
        if not pregunta_limpia:
            st.warning("Por favor ingresa una pregunta antes de generar la respuesta.")
        elif not contexto_limpio:
            st.warning("Por favor ingresa el contexto clínico.")
        else:
            with st.spinner("Generando respuesta… esto puede tardar unos segundos en CPU."):
                try:
                    respuesta, n_tokens = generar_respuesta_qa(
                        pregunta_limpia,
                        contexto_limpio,
                        tokenizador,
                        modelo,
                        dispositivo,
                        params_qa,
                    )
                    st.success("✅ Respuesta generada correctamente.")
                    st.markdown("### 💬 Respuesta generada")
                    st.info(respuesta)
                    st.caption(
                        f"📊 Tokens del prompt utilizados: **{n_tokens}** / 512 "
                        f"(pregunta + contexto)"
                    )
                    if n_tokens >= 512:
                        st.warning(
                            "⚠️ El prompt fue truncado a 512 tokens. "
                            "Considera acortar el contexto para mejores resultados."
                        )
                    st.markdown(
                        "**Prompt enviado al modelo:**\n"
                        f"```\nquestion: {pregunta_limpia} context: {contexto_limpio[:200]}…\n```"
                    )
                except Exception as exc:
                    st.error(f"❌ Error durante la inferencia: {exc}")


# ═════════════════════════════════════════════════════════════════════════════
# PESTAÑA 3 — Demo comparativa
# ═════════════════════════════════════════════════════════════════════════════
with tab_comparativa:
    st.subheader("⚖️ Demo comparativa — Greedy Search vs. Beam Search")
    st.markdown(
        "Ingresa un texto clínico y genera la salida con **ambas estrategias** "
        "simultáneamente para comparar diferencias en completitud, coherencia "
        "y fidelidad factual de los resúmenes generados."
    )

    col5, col6 = st.columns([1, 1])
    with col5:
        if st.button("💡 Cargar ejemplo", key="ejemplo_comp"):
            st.session_state["input_comp"] = EJEMPLO_REPORTE
    with col6:
        if st.button("🗑️ Limpiar", key="limpiar_comp"):
            st.session_state["input_comp"] = ""

    texto_comp = st.text_area(
        "Texto clínico / radiológico (en inglés)",
        height=200,
        placeholder="Pega aquí el texto clínico en inglés para comparar estrategias…",
        key="input_comp",
    )

    if st.button("🔬 Comparar estrategias", type="primary", key="btn_comp"):
        texto_comp_limpio = texto_comp.strip()
        if not texto_comp_limpio:
            st.warning("Por favor ingresa un texto antes de comparar.")
        else:
            col_gs, col_bs = st.columns(2)
            with col_gs:
                st.markdown("### 🎯 Greedy Search")
                with st.spinner("Generando con Greedy Search…"):
                    try:
                        salida_gs, n_tok_gs = generar_resumen(
                            texto_comp_limpio,
                            tokenizador,
                            modelo,
                            dispositivo,
                            params=GREEDY_PARAMS,
                        )
                        st.success("✅ Completado")
                        st.info(salida_gs)
                        st.caption(f"Tokens entrada: **{n_tok_gs}**")
                    except Exception as exc:
                        st.error(f"❌ Error: {exc}")

            with col_bs:
                st.markdown("### 🎯 Beam Search")
                with st.spinner("Generando con Beam Search…"):
                    try:
                        salida_bs, n_tok_bs = generar_resumen(
                            texto_comp_limpio, tokenizador, modelo, dispositivo
                        )
                        st.success("✅ Completado")
                        st.info(salida_bs)
                        st.caption(f"Tokens entrada: **{n_tok_bs}**")
                    except Exception as exc:
                        st.error(f"❌ Error: {exc}")

            st.markdown("---")
            st.markdown(
                "**Notas de interpretación:**\n\n"
                "**Greedy Search** — selecciona en cada paso el token de mayor probabilidad "
                "(argmax local). Es el método más simple y rápido, útil como baseline. Su "
                "principal desventaja es que queda atrapado en óptimos locales: al no explorar "
                "alternativas, puede omitir hallazgos clínicos relevantes y producir resúmenes "
                "menos completos o con estructura menos coherente. Para textos clínicos, esto "
                "se traduce en resúmenes que priorizan los primeros hallazgos y omiten los menos "
                "inmediatos, aunque sean críticos.\n\n"
                "**Beam Search** — mantiene simultáneamente varias hipótesis (num_beams=4) y "
                "maximiza la probabilidad conjunta de toda la secuencia generada. Para resúmenes "
                "clínicos y radiológicos, donde la completitud factual y la fidelidad al reporte "
                "son críticas, Beam Search es la estrategia preferida: al explorar múltiples "
                "caminos en paralelo, encuentra secuencias con mayor probabilidad global que "
                "incluyen más hallazgos relevantes. Su desventaja es el mayor costo computacional "
                "y una tendencia a generar salidas más conservadoras y predecibles."
            )

# ─────────────────────────────────────────────────────────────────────────────
# Pie de página
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "🎓 Demo académica — BioBART (Yuan et al., 2022) · "
    "Procesamiento de Datos Secuenciales · "
    "Modelo: `hamzamalik11/Biobart_radiology_summarization` · "
    "No apto para uso clínico real."
)
