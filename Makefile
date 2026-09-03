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

fetch-models:  ## Download required MediaPipe task bundles
	@mkdir -p models/mediapipe
	@echo "Fetching MediaPipe hand_landmarker.task ..."
	@curl -sL -o models/mediapipe/hand_landmarker.task \
		https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
	@echo "Fetching MediaPipe pose_landmarker (sign-language track) ..."
	@curl -sL -o models/mediapipe/pose_landmarker_lite.task \
		https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
	@echo "Fetching MediaPipe face_landmarker (non-manual markers) ..."
	@curl -sL -o models/mediapipe/face_landmarker.task \
		https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
	@ls -lh models/mediapipe/

extract-asl:  ## Extract landmarks from the ASL alphabet dataset
	python -m training.datasets.extract_landmarks \
		--parquet-dir data/raw/asl_parquet --output data/sequences/asl_alphabet

extract-hagrid:  ## Extract landmarks from HaGRID parquet shards
	python -m training.datasets.extract_landmarks \
		--parquet-dir data/raw/hagrid_parquet --output data/sequences/hagrid --max-per-class 4000

# Replaces models/asl_alphabet, whose current training data carries no licence.
# ASL-HG is CC BY 4.0: commercial use allowed, attribution only, no share-alike.
# Download ASL_Raw_Images.zip by hand first - Mendeley does not serve files to
# the API - and unzip it so that data/raw/asl_hg/ holds one folder per class:
#     https://data.mendeley.com/datasets/j4y5w2c8w9/1
# Class "0" is excluded because ASL-HG uses the two-handed sign for it and this
# pipeline encodes a single hand; see docs/DATASET_LICENSES.md.
retrain-asl-clean:  ## Rebuild the fingerspelling model on CC BY 4.0 data (ASL-HG)
	@test -d data/raw/asl_hg || { \
		echo "error: data/raw/asl_hg not found."; \
		echo "Download ASL_Raw_Images.zip from https://data.mendeley.com/datasets/j4y5w2c8w9/1"; \
		echo "and unzip it to data/raw/asl_hg/ (one folder per class)."; exit 1; }
	python -m training.datasets.extract_landmarks \
		--image-dir data/raw/asl_hg --output data/sequences/asl_hg --exclude-classes 0
	python dextera.py train --dataset data/sequences/asl_hg \
		--epochs 80 --calibrate --export models/asl_alphabet
	cd apps/web && npm run sync-runtime
	@echo "Done. Update docs/DATASET_LICENSES.md and models/LICENSE to match."

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
