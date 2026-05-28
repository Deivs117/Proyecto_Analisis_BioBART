# ─────────────────────────────────────────────────────────────────────────────
# Makefile — BioBART Streamlit App
# Uso: desde la carpeta streamlit_app/ ejecutar `make <comando>`
# ─────────────────────────────────────────────────────────────────────────────

PYTHON    := python3
UV        := uv
VENV      := .venv
VENV_BIN  := $(VENV)/bin
APP       := app.py
PORT      := 8501

# Detecta el activador según OS
ifeq ($(OS),Windows_NT)
    ACTIVATE := $(VENV)/Scripts/activate
else
    ACTIVATE := $(VENV_BIN)/activate
endif

.DEFAULT_GOAL := help

# ─────────────────────────────────────────────────────────────────────────────
# help — muestra los comandos disponibles
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "  BioBART Streamlit App — Comandos disponibles"
	@echo "  ─────────────────────────────────────────────"
	@echo "  make setup   → Crea el entorno virtual e instala dependencias"
	@echo "  make run     → Lanza la app en http://localhost:$(PORT)"
	@echo "  make all     → setup + run (todo en un solo comando)"
	@echo "  make lint    → Ejecuta ruff para verificar el código"
	@echo "  make clean   → Elimina el entorno virtual y artefactos"
	@echo ""

# ─────────────────────────────────────────────────────────────────────────────
# setup — crea el entorno virtual e instala dependencias con uv
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: setup
setup:
	@echo ">>> Creando entorno virtual con uv..."
	$(UV) venv $(VENV)
	@echo ">>> Instalando dependencias..."
	$(UV) pip install --python $(VENV_BIN)/python -e .
	@echo ""
	@echo "✅ Entorno listo. Ejecuta: make run"

# ─────────────────────────────────────────────────────────────────────────────
# run — lanza la app Streamlit
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: run
run:
	@echo ">>> Iniciando BioBART Streamlit App en http://localhost:$(PORT) ..."
	$(VENV_BIN)/streamlit run $(APP) \
		--server.port $(PORT) \
		--server.headless false

# ─────────────────────────────────────────────────────────────────────────────
# all — atajo para setup + run
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: all
all: setup run

# ─────────────────────────────────────────────────────────────────────────────
# lint — verifica el código con ruff
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: lint
lint:
	$(VENV_BIN)/ruff check app.py inference.py model_loader.py config.py

# ─────────────────────────────────────────────────────────────────────────────
# clean — elimina entorno virtual y artefactos de Python
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: clean
clean:
	@echo ">>> Eliminando entorno virtual y artefactos..."
	rm -rf $(VENV) __pycache__ *.egg-info .ruff_cache
	@echo "✅ Limpieza completada."
