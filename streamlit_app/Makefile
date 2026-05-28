# ─────────────────────────────────────────────────────────────────────────────
# Makefile - BioBART Streamlit App
# Uso: desde la carpeta streamlit_app/ ejecutar `make <comando>`
# ─────────────────────────────────────────────────────────────────────────────

UV        := uv
APP       := app.py
PORT      := 8501

.DEFAULT_GOAL := help

# ─────────────────────────────────────────────────────────────────────────────
# help - muestra los comandos disponibles
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "  BioBART Streamlit App - Comandos disponibles"
	@echo "  ─────────────────────────────────────────────"
	@echo "  make setup   -> Crea el entorno virtual e instala dependencias"
	@echo "  make add pkg=<paquete> -> Agrega una dependencia solo si es necesario"
	@echo "  make run     -> Lanza la app en http://localhost:$(PORT)"
	@echo "  make all     -> setup + run (todo en un solo comando)"
	@echo "  make lint    -> Ejecuta ruff para verificar el codigo"
	@echo "  make clean   -> Elimina el entorno virtual y artefactos"
	@echo ""

# ─────────────────────────────────────────────────────────────────────────────
# setup - crea el entorno virtual e instala dependencias con uv
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: setup
setup:
	@echo ">>> Sincronizando entorno con uv sync (incluye dependencias de desarrollo, sin instalar paquete local)..."
	$(UV) sync --extra dev --no-install-project
	@echo ""
	@echo "OK Entorno listo. Ejecuta: make run"

# ─────────────────────────────────────────────────────────────────────────────
# add - agrega dependencias solo cuando sea necesario
# Uso: make add pkg=<nombre-paquete>
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: add
add:
	@if [ -z "$(pkg)" ]; then \
		echo "ERROR Debes indicar el paquete: make add pkg=<nombre-paquete>"; \
		exit 1; \
	fi
	@echo ">>> Agregando dependencia con uv add..."
	$(UV) add $(pkg)
	@echo "OK Dependencia agregada. Ejecuta: make setup"

# ─────────────────────────────────────────────────────────────────────────────
# run - lanza la app Streamlit
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: run
run:
	@echo ">>> Iniciando BioBART Streamlit App en http://localhost:$(PORT) ..."
	$(UV) run --no-project streamlit run $(APP) \
		--server.port $(PORT) \
		--server.headless false \
		--server.fileWatcherType none

# ─────────────────────────────────────────────────────────────────────────────
# all - atajo para setup + run
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: all
all: setup run

# ─────────────────────────────────────────────────────────────────────────────
# lint - verifica el codigo con ruff
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: lint
lint:
	$(UV) run --no-project ruff check app.py inference.py model_loader.py config.py

# ─────────────────────────────────────────────────────────────────────────────
# clean - elimina entorno virtual y artefactos de Python
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: clean
clean:
	@echo ">>> Eliminando entorno virtual y artefactos..."
	rm -rf .venv __pycache__ *.egg-info .ruff_cache
	@echo "OK Limpieza completada."
