# 🚀 Proyecto Final PLN: BioBART para Generación de Texto Biomédico

**Materia:** Procesamiento de Datos Secuenciales  
**Módulo:** Procesamiento de Lenguaje Natural (PLN)

Este repositorio contiene la implementación y el material de análisis para estudiar la arquitectura **Encoder-Decoder** de los LLMs, basada en *Attention Is All You Need*, con foco en **BART** y su adaptación al dominio biomédico: **BioBART**.

El objetivo principal es comprender en profundidad el flujo de inferencia (entradas/salidas), los mecanismos de atención (**Self-Attention, Masked Self-Attention y Cross-Attention**) y justificar por qué en tareas biomédicas es crítico **generar texto nuevo** y no solo comprenderlo.

---

## 📁 Estructura del Proyecto

```text
Proyecto_Analisis_BioBART/
├── docs/
│   └── guia_llm_biobart.html
├── notebooks/
│   ├── biobart_inferencia.ipynb
│   └── README.md
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

- `docs/`: guía teórica visual e interactiva en HTML para repaso técnico.
- `notebooks/`: notebook de inferencia comentada y material de apoyo para la exposición.
- `.gitignore`: configuración para excluir artefactos de Python/Jupyter y archivos temporales.
- `pyproject.toml`: dependencias y metadatos del proyecto gestionados con `uv`.
- `requirements.txt`: lista de dependencias de respaldo (referencia).

---

## 🧠 Resumen técnico: BART/BioBART y la problemática biomédica

### ¿Qué arquitectura estudiamos?
- **BART** es un modelo híbrido **Encoder-Decoder**.
- El **Encoder** contextualiza toda la secuencia de entrada con atención bidireccional.
- El **Decoder** genera texto de forma autoregresiva usando:
  1. **Masked Self-Attention** (sin ver tokens futuros),
  2. **Cross-Attention** (consulta la memoria del encoder),
  3. **Feed Forward + Normalización + Residuales**.

### ¿Por qué BioBART?
- **BioBART** adapta BART al lenguaje biomédico especializado.
- En biomedicina, no basta con clasificar o extraer información: muchas tareas exigen **redacción** (resúmenes clínicos, síntesis de literatura, reportes).
- Un enfoque solo-encoder (como BERT) es fuerte para comprensión, pero no está diseñado para generación secuencial completa.

---

## ⚙️ Instalación y ejecución del notebook (con `uv`)

### 1) Instalar `uv` (una sola vez)
```bash
curl -Lsf https://astral.sh/uv/install.sh | sh
```

### 2) Clonar e ingresar al repositorio
```bash
git clone https://github.com/Deivs117/Proyecto_Analisis_BioBART.git
cd Proyecto_Analisis_BioBART
```

### 3) Crear entorno virtual e instalar dependencias
```bash
uv venv .venv
source .venv/bin/activate          # En Windows: .venv\Scripts\activate
uv pip install -e .
```

### 4) Registrar el kernel de Jupyter y ejecutar JupyterLab
```bash
python -m ipykernel install --user --name biobart-env --display-name "BioBART (uv)"
jupyter lab
```
Luego abre `notebooks/biobart_inferencia.ipynb` y selecciona el kernel **"BioBART (uv)"**.

---

## 🧾 Guía exacta de comandos Git (inicio y primer push a `main`)

> Ejecuta esta secuencia desde la carpeta del proyecto local.

```bash
# 1) Inicializar repositorio
git init

# 2) Definir rama principal
git branch -M main

# 3) Conectar remoto de GitHub
git remote add origin https://github.com/Deivs117/Proyecto_Analisis_BioBART.git

# 4) Agregar todos los archivos (docs HTML, notebooks, etc.)
git add .

# 5) Primer commit
git commit -m "feat: estructura inicial del proyecto BioBART (docs + notebooks + setup)"

# 6) Subir a GitHub (rama main)
git push -u origin main
```

---

## 📚 Recurso teórico principal

- Guía visual HTML: `docs/guia_llm_biobart.html`
