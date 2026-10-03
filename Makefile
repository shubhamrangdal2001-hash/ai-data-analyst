.PHONY: help install dev-api dev-frontend dev test lint build push deploy clean

PYTHON     := python3
PIP        := pip
IMAGE_NAME := ai-data-analyst
TAG        := latest

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Development ───────────────────────────────────────────────────────────────

install: ## Install Python dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

dev-api: ## Run FastAPI backend in dev mode
	PYTHONPATH=. uvicorn backend.main:app --reload --port 8000 --log-level debug

dev-frontend: ## Run Streamlit frontend in dev mode
	PYTHONPATH=. streamlit run frontend/app.py --server.port 8501

dev: ## Run both services in dev mode (requires tmux or two terminals)
	@echo "Start in two terminals:"
	@echo "  make dev-api"
	@echo "  make dev-frontend"

generate-sample: ## Generate sample dataset
	cd data/sample && $(PYTHON) generate_sample.py

# ── Testing ───────────────────────────────────────────────────────────────────

test: ## Run all tests
	PYTHONPATH=. pytest tests/ -v --tb=short

test-unit: ## Run unit tests only
	PYTHONPATH=. pytest tests/unit/ -v

test-integration: ## Run integration tests only
	PYTHONPATH=. pytest tests/integration/ -v

test-cov: ## Run tests with coverage report
	PYTHONPATH=. pytest tests/ --cov=backend --cov-report=html --cov-report=term

lint: ## Lint with ruff
	ruff check backend/ frontend/ tests/

lint-fix: ## Auto-fix lint issues
	ruff check --fix backend/ frontend/ tests/

# ── Docker ────────────────────────────────────────────────────────────────────

build: ## Build Docker image
	docker build -t $(IMAGE_NAME):$(TAG) .

build-no-cache: ## Build Docker image (no cache)
	docker build --no-cache -t $(IMAGE_NAME):$(TAG) .

up: ## Start with Docker Compose
	docker-compose up --build

up-d: ## Start with Docker Compose (detached)
	docker-compose up --build -d

down: ## Stop Docker Compose
	docker-compose down

logs: ## View Docker Compose logs
	docker-compose logs -f

# ── Azure ─────────────────────────────────────────────────────────────────────

azure-deploy: ## Deploy to Azure (requires env vars set)
	chmod +x deployment/azure/deploy.sh
	./deployment/azure/deploy.sh

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean: ## Remove temp files and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null; true
	find . -name "htmlcov" -exec rm -rf {} + 2>/dev/null; true
	rm -rf data/uploads/* data/reports/* logs/*.log

.DEFAULT_GOAL := help
