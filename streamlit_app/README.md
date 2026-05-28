# 🏥 BioBART — Interfaz Web de Demo para Sustentación Académica

Aplicativo web interactivo construido con **Streamlit** que expone las funcionalidades de inferencia del modelo [BioBART](https://arxiv.org/abs/2204.03905) demostradas en el notebook principal del proyecto.

> **Materia:** Procesamiento de Datos Secuenciales  
> **Módulo:** Procesamiento de Lenguaje Natural (PLN)

---

## 📋 Descripción

Esta app permite demostrar, de forma visual e interactiva, las tres capacidades principales de BioBART para textos clínicos y radiológicos:

| Modo | Tarea | Estrategia de generación |
|------|-------|-------------------------|
| **Resumen radiológico** | Resume un reporte clínico largo | Beam Search (`num_beams=4`) |
| **QA médico** | Responde preguntas dado un contexto clínico | Beam Search (`num_beams=4`) |
| **Demo comparativa** | Compara ambas estrategias sobre el mismo texto | Greedy Search vs. Beam Search en paralelo |

---

## 📁 Estructura de archivos

```text
streamlit_app/
├── app.py              ← Interfaz Streamlit principal (UI)
├── inference.py        ← Funciones de inferencia (resumen y QA)
├── model_loader.py     ← Carga del modelo con caché de Streamlit
├── config.py           ← Constantes, parámetros e hiperparámetros por defecto
├── Makefile            ← Comandos simples para setup y ejecución
├── pyproject.toml      ← Dependencias gestionadas con uv
├── .gitignore
└── README.md           ← Este archivo
```

---

## ⚙️ Dependencias

| Paquete | Versión mínima | Propósito |
|---------|---------------|-----------|
| `streamlit` | ≥ 1.35.0 | Interfaz web |
| `transformers` | ≥ 4.40.0 | Carga de modelo BioBART y tokenizador |
| `torch` | ≥ 2.2.0 | Backend de inferencia (CPU/GPU) |
| `sentencepiece` | ≥ 0.1.99 | Tokenizador BPE requerido por BioBART |
| `accelerate` | ≥ 0.28.0 | Optimización de inferencia |

---

## 🚀 Instalación y ejecución

### Prerrequisito: instalar `uv`

```bash
curl -Lsf https://astral.sh/uv/install.sh | sh
```

### Opción A — Un solo comando (recomendado para sustentación)

```bash
cd streamlit_app/
make all
```

Este comando:
1. Crea el entorno virtual `.venv/`
2. Instala todas las dependencias con `uv`
3. Lanza la app en `http://localhost:8501`

### Opción B — Comandos separados

```bash
cd streamlit_app/
make setup   # crea entorno e instala dependencias
make run     # lanza la app
```

### Opción C — Manual (sin Make)

```bash
cd streamlit_app/

# Crear entorno virtual
uv venv .venv

# Activar entorno
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows PowerShell

# Instalar dependencias
uv pip install -e .

# Lanzar app
streamlit run app.py
```

---

## 📖 Descripción de cada modo

### 1. 📄 Resumen Radiológico

- **Entrada:** texto libre de un reporte clínico o radiológico (en inglés)
- **Proceso:** tokenización BPE → encoder BioBART → Beam Search decoder
- **Salida:** resumen abstractivo del reporte
- **Parámetros ajustables desde la UI:**

| Parámetro | Valor por defecto | Descripción |
|-----------|:-----------------:|-------------|
| `max_new_tokens` | 150 | Longitud máxima del resumen |
| `num_beams` | 4 | Número de hipótesis de Beam Search |
| `min_length` | 30 | Longitud mínima del resumen |
| `no_repeat_ngram_size` | 3 | Evita repetición de n-gramas |
| `length_penalty` | 2.0 | Penaliza secuencias cortas (>1 favorece largo) |
| `early_stopping` | True | Detiene cuando todos los beams llegan a EOS |

### 2. ❓ Pregunta / Respuesta Médica

- **Entrada:** pregunta clínica + contexto radiológico (en inglés)
- **Prompt:** `question: <pregunta> context: <contexto>`
- **Proceso:** concatenación estructurada → tokenización → Beam Search decoder
- **Salida:** respuesta generada de forma abstractiva
- **Parámetros ajustables desde la UI:**

| Parámetro | Valor por defecto | Descripción |
|-----------|:-----------------:|-------------|
| `max_new_tokens` | 200 | Longitud máxima de la respuesta |
| `num_beams` | 4 | Número de hipótesis de Beam Search |
| `min_length` | 10 | Longitud mínima de la respuesta |
| `no_repeat_ngram_size` | 3 | Evita repetición de n-gramas |

### 3. ⚖️ Demo Comparativa

- Genera la salida del mismo texto con **Greedy Search** y **Beam Search** en paralelo
- Útil para demostrar diferencias en completitud factual y coherencia durante la sustentación

---

## 🧠 Modelo utilizado

**`GanjinZero/biobart-base`** — BioBART base (~560 MB)

- Pre-entrenado sobre abstracts de **PubMed Central Open Access**
- Arquitectura: BART encoder-decoder
  - Encoder: 6 capas, 768 dimensiones, 12 cabezas de atención
  - Decoder: 6 capas, 768 dimensiones, 12 cabezas de atención
- Vocabulario BPE: ~50,264 tokens biomédicos
- Ventana máxima: **1024 tokens** (configurado a 512 en esta demo)
- Publicado en: [arxiv.org/abs/2204.03905](https://arxiv.org/abs/2204.03905)

### Cambiar a BioBART-large

Editar `config.py`:

```python
MODEL_NAME = "GanjinZero/biobart-large"  # ~1.6 GB, requiere más RAM/VRAM
```

---

## ⚡ Optimización de carga del modelo

El modelo se carga **una sola vez** gracias a `@st.cache_resource`:

```python
@st.cache_resource(show_spinner="Cargando modelo…")
def cargar_modelo():
    tokenizador = BartTokenizer.from_pretrained(MODEL_NAME)
    modelo = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    modelo.to(dispositivo)
    modelo.eval()
    return tokenizador, modelo, dispositivo
```

La primera carga puede tardar **2–5 minutos** en CPU (descarga del modelo ~560 MB).  
Las interacciones posteriores son instantáneas.

---

## ⚠️ Advertencias y limitaciones

1. **Idioma:** el modelo fue pre-entrenado en inglés. Los textos en español producirán resultados degradados.
2. **Truncamiento:** reportes superiores a 512 tokens serán truncados automáticamente.
3. **Alucinaciones:** el modelo puede generar valores de laboratorio o dosis plausibles pero incorrectas.
4. **Uso clínico:** esta es una **demo académica**. No debe usarse para diagnóstico o tratamiento real.
5. **Velocidad:** en CPU, cada inferencia puede tardar 20–60 segundos dependiendo del hardware.

---

## 🔧 Otros comandos Make

```bash
make lint    # Verifica el código con ruff
make clean   # Elimina el entorno virtual y artefactos
make help    # Muestra todos los comandos disponibles
```

---

## 📝 Notas para sustentación

1. **Antes de la sustentación:** ejecutar `make setup` con antelación para descargar el modelo (~560 MB).
2. **Durante la sustentación:** usar `make run` para iniciar la app ya con el modelo en caché.
3. **Ejemplos de texto:** la app incluye botones "💡 Cargar ejemplo" en cada modo para no tener que escribir texto durante la demo.
4. **Parámetros:** los controles deslizantes permiten demostrar en tiempo real el efecto de cada hiperparámetro.
5. **Comparativa:** la pestaña "⚖️ Demo Comparativa" es ideal para explicar diferencias entre Greedy Search y Beam Search.

---

## 🔗 Referencias

- Yuan, Z. et al. (2022). *BioBART: Pretraining and Evaluation of A Biomedical Generative Language Model.* [arxiv.org/abs/2204.03905](https://arxiv.org/abs/2204.03905)
- Lewis, M. et al. (2019). *BART: Denoising Sequence-to-Sequence Pre-training.* [arxiv.org/abs/1910.13461](https://arxiv.org/abs/1910.13461)
- Modelo en HuggingFace: [GanjinZero/biobart-base](https://huggingface.co/GanjinZero/biobart-base)
