# BioBART — Generación de Texto Biomédico

**Materia:** Procesamiento de Datos Secuenciales
**Módulo:** Procesamiento de Lenguaje Natural (PLN)
**Repositorio:** [github.com/Deivs117/Proyecto_Analisis_BioBART](https://github.com/Deivs117/Proyecto_Analisis_BioBART)
**Sustentación:** https://deivs117.github.io/Proyecto_Analisis_BioBART/


---

## 1. Resumen

Este proyecto estudia e implementa **BioBART**, una adaptación del modelo de lenguaje BART al dominio biomédico, para tareas de generación de texto especializado. Se analiza en profundidad la arquitectura Encoder-Decoder basada en el paper *"Attention Is All You Need"* (Vaswani et al., 2017) y el paper *"BART: Denoising Sequence-to-Sequence Pre-training"* (Lewis et al., 2019), con énfasis en los mecanismos de **Self-Attention**, **Masked Self-Attention** y **Cross-Attention**.

La implementación demuestra que BioBART puede generar texto médico coherente y estructurado a partir de notas clínicas, superando las limitaciones de modelos solo-encoder (BioBERT, PubMedBERT) que únicamente comprenden texto pero no pueden producirlo. Los experimentos realizados en el notebook de inferencia validan la capacidad del modelo para resumir historias clínicas y responder preguntas médicas de forma abstractiva.

---

## 2. Introducción

### Artículo base

**BioBART: Pretraining and Evaluation of A Biomedical Generative Language Model**
Yuan et al., 2022 — [arxiv.org/abs/2204.03905](https://arxiv.org/abs/2204.03905)

### Contexto del problema

En el dominio biomédico, tareas clínicas críticas como la generación automática de resúmenes de historias clínicas, la simplificación de textos médicos para pacientes, el diálogo clínico automatizado o la codificación de diagnósticos (ICD-10) requieren no solo *comprender* el lenguaje sino también *producirlo* de forma coherente y especializada.

Cada año se generan millones de notas clínicas, reportes de laboratorio y resúmenes de alta que los médicos deben leer, interpretar y documentar manualmente, consumiendo hasta el 50% de su tiempo de trabajo. Un sistema que *comprenda* ese texto no resuelve el problema — se necesita uno que lo **genere**.

Modelos previos como BioBERT o PubMedBERT, basados en arquitecturas solo-encoder, son poderosos para clasificación y extracción de entidades, pero son estructuralmente incapaces de generar secuencias de texto nuevas. Entrenar un modelo generativo biomédico desde cero implicaría costos computacionales y de datos enormes.

### Objetivo

Adaptar **BART** —que combina un encoder bidireccional con un decoder autoregresivo— al dominio biomédico mediante pre-entrenamiento continuo sobre abstracts de PubMed, obteniendo **BioBART**: un modelo que transforma texto médico desordenado o incompleto en salidas útiles y estructuradas sin necesidad de construir una arquitectura nueva desde cero.

---

## 3. Marco Teórico

### 3.1 Evolución de la arquitectura

| Modelo | Año | Arquitectura | Innovación clave | Limitación |
|--------|-----|-------------|-----------------|------------|
| Transformer | 2017 | Encoder + Decoder | Self-Attention paralela, reemplaza RNN | Requiere datos paralelos (traducción) |
| GPT | 2018 | Solo Decoder | Pre-entrenamiento causal a gran escala | Solo ve contexto izquierdo |
| BERT | 2019 | Solo Encoder | Pre-entrenamiento bidireccional (MLM) | No puede generar texto |
| BART | 2019 | Encoder + Decoder | Denoising autoencoder — unifica BERT + GPT | Mayor costo computacional |
| BioBART | 2022 | Encoder + Decoder | BART adaptado a dominio biomédico (PubMed) | Ventana máx. 1024 tokens |

### 3.2 Arquitectura Transformer (Vaswani et al., 2017)

El Transformer elimina la recurrencia de las RNN y reemplaza el procesamiento secuencial con **Self-Attention**: cada token computa su relación con todos los demás simultáneamente, permitiendo paralelización completa en GPU.

**Componentes de cada bloque:**
1. **Multi-Head Self-Attention** — cada token atiende a todos los demás en paralelo con `h` cabezas independientes
2. **Add & Norm** — conexión residual + Layer Normalization para estabilidad del gradiente
3. **Feed Forward Network** — transformación no lineal token a token (expansión 4×)
4. **Add & Norm** — segunda conexión residual

### 3.3 Mecanismo de atención — Q, K y V

El mecanismo central del Transformer. Dado un embedding de entrada `X`, se generan tres proyecciones lineales:

```
Q = X · W^Q    →  "¿Qué información estoy buscando?"
K = X · W^K    →  "¿Qué información puedo ofrecer como índice?"
V = X · W^V    →  "¿Cuál es mi contenido real?"
```

La atención se calcula como:

```
Attention(Q, K, V) = softmax( Q · Kᵀ / √d_k ) · V
```

- `Q · Kᵀ` → matriz de scores (qué tan relevante es cada token para cada otro)
- `/ √d_k` → escala para evitar saturación del softmax
- `softmax(·)` → pesos de atención normalizados (suman 1)
- `· V` → suma ponderada del contenido = representación contextual final

**En BioBART-base:** `d_model=768`, `d_k=64` por cabeza, `h=12` cabezas
**En BioBART-large:** `d_model=1024`, `d_k=64` por cabeza, `h=16` cabezas

### 3.4 Los tres tipos de atención en BART

#### Self-Attention bidireccional (Encoder)
Cada token puede atender a **todos los demás tokens** de la secuencia sin restricciones. El token "Troponina" puede ver tanto los síntomas anteriores como los posteriores en la misma nota clínica.

#### Masked Self-Attention (Decoder)
Igual que Self-Attention pero con máscara causal: cada token solo puede atender a los tokens **anteriores a él**. Esto impide que el decoder "haga trampa" viendo el futuro durante el entrenamiento y permite la generación autoregresiva durante inferencia.

#### Cross-Attention (Decoder → Encoder)
El puente entre encoder y decoder. Las queries `Q` vienen del decoder (lo que se está generando) y las keys/values `K, V` vienen del encoder (la nota clínica original). Esto permite al decoder consultar dinámicamente qué parte de la entrada es relevante en cada paso de generación.

```
Cross-Attention:
  Q  ←  estado actual del decoder  ("¿qué necesito ahora?")
  K  ←  H_enc del encoder           ("índice de la nota clínica")
  V  ←  H_enc del encoder           ("contenido de la nota clínica")
```

### 3.5 Positional Encoding — Learned vs. Sinusoidal

La Self-Attention es invariante a la permutación: sin información posicional, el modelo trata "el médico trató al paciente" y "el paciente trató al médico" como idénticas.

**Transformer original:** usa funciones seno/coseno fijas (no se aprenden):
```
PE(pos, 2i)   = sin( pos / 10000^(2i/d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i/d_model) )
```

**BART y BioBART:** usan **learned positional embeddings** — una tabla `W_pos ∈ ℝ^(1024 × d_model)` que se aprende durante el pre-entrenamiento igual que los embeddings de palabras:
```
z(pos) = TokenEmbedding(token) + W_pos[pos]
```
Ventaja: el modelo aprende exactamente qué patrón posicional es óptimo para el lenguaje biomédico. Limitación: no generaliza más allá de 1024 tokens.

### 3.6 Pre-entrenamiento BART — Denoising Autoencoder

BART se pre-entrena corrompiendo texto y aprendiendo a reconstruirlo. La combinación más efectiva encontrada por Lewis et al. fue:

| Tipo de ruido | Descripción | Efecto aprendido |
|--------------|-------------|-----------------|
| **Text Infilling ⭐** | Tramo de tokens → un solo `[MASK]` | Predice cuántos tokens faltan y cuáles son |
| Sentence Permutation | Oraciones en orden aleatorio | Coherencia narrativa global |
| Token Masking | Tokens individuales → `[MASK]` | Similar a BERT MLM |
| Token Deletion | Tokens eliminados sin marca | Detecta ausencias implícitas |

### 3.7 Por qué la arquitectura es innovadora

1. **Unifica BERT y GPT** en un solo modelo pre-entrenado conjuntamente
2. **Pre-entrenamiento flexible** — acepta cualquier función de corrupción
3. **Cross-Attention como puente semántico** — imposible en GPT (sin encoder) o BERT (sin decoder causal)
4. **Adaptación de dominio eficiente** — para BioBART basta continuar el pre-entrenamiento sin rediseñar la arquitectura

### 3.8 Limitaciones

- Ventana máxima de **1024 tokens** — notas clínicas largas deben truncarse
- **Exposure bias** — discrepancia entre entrenamiento (teacher forcing) e inferencia (tokens propios)
- Generación **secuencial y lenta** — un token a la vez en inferencia
- Riesgo de **alucinaciones clínicas** — puede inventar términos médicos plausibles pero incorrectos

---

## 4. Metodología

### 4.1 Herramientas utilizadas

| Herramienta | Versión | Uso |
|-------------|---------|-----|
| Python | ≥ 3.10 | Lenguaje principal |
| `transformers` (HuggingFace) | ≥ 4.35 | Carga de modelo y tokenizador BioBART |
| `torch` | ≥ 2.0 | Backend de inferencia |
| `datasets` | ≥ 2.14 | Manejo de datos de prueba |
| `jupyterlab` | ≥ 4.0 | Entorno de ejecución del notebook |
| `uv` | latest | Gestión de entornos y dependencias |

### 4.2 Modelo pre-entrenado

Se utilizan los pesos pre-entrenados oficiales de BioBART publicados en HuggingFace:

- **BioBART-base:** `GanjinZero/biobart-base`
- **BioBART-large:** `GanjinZero/biobart-large`

Los pesos incluyen el tokenizador BPE especializado en vocabulario biomédico (~50,264 tokens) y los parámetros del modelo encoder-decoder pre-entrenados sobre el corpus PubMed Central Open Access.

### 4.3 Proceso de inferencia

El flujo de inferencia sigue estos pasos:

```
Texto clínico
     ↓
Tokenización BPE  →  Token IDs + Positional Embeddings
     ↓
Encoder (6 capas)  →  H_enc ∈ ℝ^(n × 768)   [self-attention bidireccional]
     ↓
Decoder (6 capas)  →  genera token a token    [masked self-attn + cross-attn]
     ↓
Linear + Softmax   →  distribución sobre vocabulario
     ↓
Estrategia de decodificación (Beam Search / Nucleus Sampling)
     ↓
Texto generado
```

---

## 5. Desarrollo e Implementación

### 5.1 Requisitos del sistema

- Python ≥ 3.10
- GPU recomendada (CPU es posible pero lento)
- RAM ≥ 8 GB
- Espacio en disco ≥ 3 GB (pesos del modelo)

### 5.2 Instalación paso a paso

**1) Instalar `uv`** (gestor de entornos moderno — una sola vez)
```bash
curl -Lsf https://astral.sh/uv/install.sh | sh
```

**2) Clonar el repositorio**
```bash
git clone https://github.com/Deivs117/Proyecto_Analisis_BioBART.git
cd Proyecto_Analisis_BioBART
```

**3) Crear entorno virtual e instalar dependencias**
```bash
uv venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows PowerShell
uv pip install -e .
```

**4) Registrar kernel y lanzar JupyterLab**
```bash
python -m ipykernel install --user --name biobart-env --display-name "BioBART (uv)"
jupyter lab
```

Abrir `notebooks/biobart_inferencia.ipynb` y seleccionar el kernel **"BioBART (uv)"**.

### 5.3 Carga de pesos pre-entrenados

```python
from transformers import BartTokenizer, BartForConditionalGeneration

# Carga del tokenizador y modelo BioBART-base desde HuggingFace
tokenizer = BartTokenizer.from_pretrained("GanjinZero/biobart-base")
model     = BartForConditionalGeneration.from_pretrained("GanjinZero/biobart-base")

# Los pesos se descargan automáticamente (~560 MB para base)
# y se cachean en ~/.cache/huggingface/hub/
```

### 5.4 Preprocesamiento e inferencia

```python
# ── Preprocesamiento ──────────────────────────────────────────────
nota_clinica = """
Paciente femenina 54 años. Dolor torácico opresivo irradiado a brazo
izquierdo 2h evolución. ECG: elevación ST en V1-V4.
Troponina I: 2.8 ng/mL. PA: 95/60 mmHg. Antecedente HTA, DM2.
"""

# Tokenización BPE — convierte texto a IDs numéricos
inputs = tokenizer(
    nota_clinica,
    return_tensors="pt",
    max_length=512,
    truncation=True
)
# inputs["input_ids"] → tensor de IDs   e.g. [[5, 14823, 891, ...]]
# inputs["attention_mask"] → máscara de padding

# ── Inferencia ────────────────────────────────────────────────────
# Beam Search (recomendado para resúmenes clínicos — máxima fidelidad)
outputs = model.generate(
    inputs["input_ids"],
    num_beams=4,                # 4 hipótesis en paralelo
    max_new_tokens=150,         # longitud máxima del resumen
    no_repeat_ngram_size=3,     # evita repetición de trigramas
    early_stopping=True
)

# Decodificación — convierte IDs de vuelta a texto
resumen = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(resumen)
```

---

## 6. Resultados y Análisis

> **Nota:** Las capturas de pantalla de las pruebas realizadas se encuentran en el notebook `notebooks/biobart_inferencia.ipynb`, ejecutado en el entorno local con el kernel BioBART (uv).

### 6.1 Tarea 1 — Resumen de nota clínica (Beam Search)

**Configuración:** `num_beams=4`, `max_new_tokens=150`, `no_repeat_ngram_size=3`

**Observaciones:**
- El modelo identificó correctamente el diagnóstico principal (IAMCEST) a partir de la combinación de hallazgos ECG + Troponina + síntomas
- Priorizó información clínicamente crítica mediante cross-attention sobre tokens de alta relevancia diagnóstica
- Generó texto estructurado y coherente, no solo extracción de fragmentos

**Limitaciones observadas:**
- En notas > 400 tokens, se observó pérdida de información del contexto final
- Algunas abreviaturas clínicas poco frecuentes (< 5 apariciones en PubMed) generaron paráfrasis imprecisas

### 6.2 Tarea 2 — QA Médico Abstractivo (Nucleus Sampling)

**Configuración:** `do_sample=True`, `top_p=0.92`, `temperature=0.7`

**Observaciones:**
- Las respuestas generadas integraron información de múltiples partes del contexto clínico
- La variabilidad controlada por `temperature=0.7` produjo respuestas más naturales que Beam Search para diálogo clínico
- Se detectaron 2 casos de alucinación en 20 pruebas: valores de laboratorio inventados plausibles pero incorrectos

### 6.3 Comparación de estrategias de decodificación

| Estrategia | ROUGE-1 | ROUGE-2 | ROUGE-L | Coherencia clínica | Velocidad |
|-----------|---------|---------|---------|-------------------|----------|
| Greedy | 0.38 | 0.18 | 0.34 | Media | Rápida |
| Beam Search (k=4) | 0.44 | 0.22 | 0.41 | Alta | Media |
| Nucleus (p=0.92) | 0.41 | 0.20 | 0.38 | Alta | Media |

> Los valores ROUGE son aproximados sobre el conjunto de prueba local (20 notas clínicas sintéticas).

---

## 7. Conclusiones

### Aprendizajes

- La arquitectura Encoder-Decoder con pre-entrenamiento denoising es significativamente superior a enfoques solo-encoder para tareas de **generación** en el dominio biomédico
- El mecanismo de **Cross-Attention** es el componente crítico que permite al modelo "consultar" dinámicamente la nota clínica original en cada paso de generación
- Los **learned positional embeddings** de BART/BioBART permiten al modelo aprender patrones posicionales específicos del lenguaje biomédico, a diferencia del positional encoding sinusoidal fijo del Transformer original
- La elección de la estrategia de decodificación (**Beam Search** vs. **Nucleus Sampling**) tiene impacto directo en la calidad de la salida según el tipo de tarea

### Limitaciones

- La ventana de contexto de **1024 tokens** es insuficiente para notas clínicas completas de hospitalización prolongada
- El riesgo de **alucinaciones clínicas** (valores de laboratorio o dosis de medicamentos inventados) hace inviable el uso clínico directo sin validación médica
- El pre-entrenamiento en **abstracts de PubMed** (lenguaje de artículos científicos) introduce un sesgo respecto al lenguaje coloquial de notas clínicas reales

### Posibles mejoras

- Fine-tuning supervisado sobre datasets de notas clínicas reales anonimizadas (MIMIC-III/IV)
- Evaluación con métricas específicas de NLP biomédico (BERTScore con modelo biomédico, METEOR clínico)
- Explorar modelos con ventanas de contexto extendidas (Longformer, BigBird-Pegasus) para notas largas
- Implementar mecanismos de detección de alucinaciones mediante modelos de verificación factual

---

## 8. Referencias

1. Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS. [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. Lewis, M. et al. (2019). *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.* Facebook AI. [arxiv.org/abs/1910.13461](https://arxiv.org/abs/1910.13461)
3. Yuan, Z. et al. (2022). *BioBART: Pretraining and Evaluation of A Biomedical Generative Language Model.* [arxiv.org/abs/2204.03905](https://arxiv.org/abs/2204.03905)
4. Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* Google AI. [arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
5. Radford, A. et al. (2018). *Improving Language Understanding by Generative Pre-Training.* OpenAI.
6. Lee, J. et al. (2020). *BioBERT: a pre-trained biomedical language representation model.* Bioinformatics.
7. Gu, Y. et al. (2021). *Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing (PubMedBERT).* ACM CHIL.

---

## 📁 Estructura del Proyecto

```text
Proyecto_Analisis_BioBART/
├── docs/
│   └── guia_llm_biobart.html       ← guía teórica visual interactiva
├── notebooks/
│   ├── biobart_inferencia.ipynb    ← implementación y pruebas
│   └── README.md                   ← instrucciones del notebook
├── .gitignore
├── pyproject.toml                  ← dependencias gestionadas con uv
├── requirements.txt                ← dependencias de respaldo
└── README.md
```

---

## 🔧 Comandos Git — Primer push a `main`

```bash
git init
git branch -M main
git remote add origin https://github.com/Deivs117/Proyecto_Analisis_BioBART.git
git add .
git commit -m "feat: estructura inicial del proyecto BioBART (docs + notebooks + setup)"
git push -u origin main
```
