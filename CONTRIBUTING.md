# Contributing to DexteraAI

## Quick Setup

```bash
git clone https://github.com/R-Priyadarshi/DexteraAI.git
cd DexteraAI
pip install -e ".[dev,training]"
pre-commit install
make test-fast
```

## Development Workflow

1. Create a branch: `git checkout -b feat/your-feature`
2. Make changes
3. Run tests: `make test`
4. Run linter: `make lint`
5. Commit with conventional commits: `feat: add new gesture class`
6. Open a PR against `main`

## Code Standards

- **Python 3.11+** — use modern syntax (`X | Y` unions, `match`, etc.)
- **Type hints everywhere** — `mypy --strict` must pass
- **Ruff** for linting and formatting
- **Docstrings** on all public classes and functions (Google style)
- **Tests** for all new functionality (pytest + hypothesis)

## Architecture Rules

1. **`core/`** has zero dependency on `training/` or `apps/`
2. **`training/`** depends on `core/` only
3. **`apps/`** depend on `core/` only
4. All protocols live in `core/types.py`
5. No hardcoded paths — use `pathlib.Path` and config objects
6. No mutable global state
7. Thread-safety for anything used in real-time pipelines

## Adding a New Gesture Class

1. Collect calibration data (minimum 100 samples)
2. Add label to `core/inference/pipeline.py:PipelineConfig.gesture_labels`
3. Add to dataset metadata
4. Retrain: `make train-full`
5. Evaluate: `python dextera.py eval --checkpoint checkpoints/best.pt --dataset data/test`
6. Export: `make export`
7. Update model card in `docs/model_card.md`

## Adding a New Platform

1. Create `apps/your_platform/`
2. Use the appropriate runtime from `core/inference/`
3. Feature extraction must match `core/landmarks/features.py` exactly
4. Add CI job in `.github/workflows/ci.yml`
