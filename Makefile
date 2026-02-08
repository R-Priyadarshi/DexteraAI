.PHONY: install dev lint test test-fast bench train export demo serve info clean help

# ─── Install ──────────────────────────────────────────────────────
install:  ## Install production dependencies
	pip install -e .

dev:  ## Install all dev dependencies
	pip install -e ".[dev,training]"
	pre-commit install

# ─── Quality ──────────────────────────────────────────────────────
lint:  ## Run linter and type checker
	ruff check core/ training/ backend/ tests/
	ruff format --check core/ training/ backend/ tests/
	mypy core/ training/ --ignore-missing-imports

format:  ## Auto-format code
	ruff format core/ training/ backend/ tests/
	ruff check --fix core/ training/ backend/ tests/

# ─── Tests ────────────────────────────────────────────────────────
test:  ## Run all tests
	pytest tests/ -v --tb=short

test-fast:  ## Run fast tests only (skip slow + gpu)
	pytest tests/ -v --tb=short -m "not slow and not gpu"

test-cov:  ## Run tests with coverage
	pytest tests/ -v --cov=core --cov=training --cov-report=html --cov-report=term

bench:  ## Run latency benchmarks
	python dextera.py benchmark --checkpoint checkpoints/best.pt --device cpu

# ─── Training ─────────────────────────────────────────────────────
train:  ## Train with synthetic data (testing)
	python dextera.py train --synthetic --epochs 10 --batch-size 32

train-full:  ## Full training run
	python dextera.py train --dataset data/gestures --epochs 100 --device auto

# ─── Export ───────────────────────────────────────────────────────
export:  ## Export best model to ONNX
	python dextera.py export --checkpoint checkpoints/best.pt --format onnx --quantize

export-all:  ## Export to ONNX + TFLite
	python dextera.py export --checkpoint checkpoints/best.pt --format both --quantize

# ─── Demo ─────────────────────────────────────────────────────────
demo:  ## Run webcam demo (detection only)
	python dextera.py demo

demo-model:  ## Run webcam demo with trained model
	python dextera.py demo --model models/gesture.onnx

# ─── API Server ───────────────────────────────────────────────────
serve:  ## Start FastAPI server (optional)
	python dextera.py serve --port 8000

info:  ## Show system information
	python dextera.py info

# ─── Web App ──────────────────────────────────────────────────────
web-install:  ## Install web app dependencies
	cd apps/web && npm install

web-dev:  ## Start web dev server
	cd apps/web && npm run dev

web-build:  ## Build web app
	cd apps/web && npm run build

# ─── Cleanup ──────────────────────────────────────────────────────
clean:  ## Clean build artifacts
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache htmlcov
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ─── Help ─────────────────────────────────────────────────────────
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
