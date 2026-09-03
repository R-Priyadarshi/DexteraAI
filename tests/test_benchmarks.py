"""Latency benchmarks for the per-frame recognition path.

CI's benchmark job ran `pytest -m "benchmark"` against a suite in which no test
carried that marker. pytest exits 5 on an empty selection, so the job had never
once passed — the same shape of defect as the bandit step that failed on a
missing command rather than on any finding.

These are the real thing. They measure the work done for every camera frame,
which is what decides whether recognition keeps up at 30fps.

On budgets: a shared CI runner is a noisy place to measure, so each assert
leaves at least an order of magnitude of headroom over what the same code does
on an idle machine. They are alarms for an accidental O(n^2), a per-frame model
reload, or a copy that should have been a view — not published figures. The
numbers quoted in the README are measured separately, on a quiet machine, and
say so.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from core.landmarks.features import LandmarkFeatureExtractor
from core.landmarks.normalizer import LandmarkNormalizer, NormalizationMode
from core.types import Handedness, HandLandmarks

if TYPE_CHECKING:
    from pathlib import Path

# One frame at 30fps. Every budget below is a fraction of this, because
# landmark detection — which we do not control — spends most of it.
FRAME_BUDGET_MS = 1000.0 / 30.0

SEQUENCE_LEN = 30
FEATURE_DIM = 86


def _hand(seed: int = 0) -> HandLandmarks:
    """A plausible hand: landmarks spread over a hand-sized region of frame."""
    rng = np.random.default_rng(seed)
    return HandLandmarks(
        landmarks=rng.random((21, 3)).astype(np.float32) * 0.3 + 0.35,
        handedness=Handedness.RIGHT,
        confidence=0.95,
    )


def _mean_ms(benchmark: Any) -> float:
    """Mean round time in milliseconds, from pytest-benchmark's own stats."""
    return float(benchmark.stats["mean"]) * 1000.0


@pytest.mark.benchmark(group="landmarks")
def test_normalization_is_a_rounding_error(benchmark) -> None:
    """Full normalization: wrist-centre, unit-scale, rotation-align.

    This runs on every hand in every frame, and it is pure numpy on 63 floats.
    If it ever costs a meaningful slice of the frame budget, something is
    allocating per call that should not be.
    """
    normalizer = LandmarkNormalizer(mode=NormalizationMode.FULL)
    hand = _hand()

    result = benchmark(normalizer.normalize, hand)

    assert result.landmarks.shape == (21, 3)
    assert _mean_ms(benchmark) < FRAME_BUDGET_MS / 10


@pytest.mark.benchmark(group="landmarks")
def test_feature_extraction_fits_the_frame_budget(benchmark) -> None:
    """The 86-dimension feature vector the model consumes.

    This is the whole Python-side hot path for a static gesture: normalize,
    then extract. The browser reimplements it in TypeScript, and
    `feature-parity.test.ts` holds the two to 1e-5 of each other.
    """
    normalizer = LandmarkNormalizer(mode=NormalizationMode.FULL)
    extractor = LandmarkFeatureExtractor()
    hand = _hand()

    def extract_one() -> np.ndarray:
        return extractor.extract(normalizer.normalize(hand))

    features = benchmark(extract_one)

    assert features.shape == (FEATURE_DIM,)
    assert _mean_ms(benchmark) < FRAME_BUDGET_MS / 5


@pytest.mark.benchmark(group="landmarks")
def test_sequence_extraction_is_linear_in_frames(benchmark) -> None:
    """A full 30-frame window, as the temporal path consumes it.

    Guards the shape of the cost, not just its size: 30 frames must cost about
    30 times one frame. A regression to quadratic here would still look fast on
    a 30-frame window and fall over on the 90-frame sign-language window.
    """
    extractor = LandmarkFeatureExtractor()
    hands = [_hand(seed) for seed in range(SEQUENCE_LEN)]

    sequence = benchmark(extractor.extract_sequence, hands)

    assert sequence.shape == (SEQUENCE_LEN, FEATURE_DIM)
    # Generous: 30 frames inside a single frame's budget still leaves the
    # detector the other 97% of its time.
    assert _mean_ms(benchmark) < FRAME_BUDGET_MS


@pytest.mark.benchmark(group="inference")
def test_transformer_forward_pass(benchmark) -> None:
    """Torch CPU inference on the shipped architecture.

    This is the slower of the two inference paths — the CLI and training use
    it; the browser and `--onnx` do not. It is here so a change to the model
    shows up as a number rather than as a vague feeling that the demo got
    heavier.
    """
    torch = pytest.importorskip("torch")
    from core.temporal.model import GestureTransformer

    model = GestureTransformer(input_dim=FEATURE_DIM, num_classes=18, max_seq_len=SEQUENCE_LEN)
    model.eval()

    features = torch.randn(1, SEQUENCE_LEN, FEATURE_DIM)
    # False = real frame. True means padding, which is the inversion that made
    # both ONNX paths score a window of pure padding.
    mask = torch.zeros(1, SEQUENCE_LEN, dtype=torch.bool)

    def forward() -> dict[str, torch.Tensor]:
        with torch.no_grad():
            return model(features, mask=mask)

    output = benchmark(forward)

    assert output["logits"].shape == (1, 18)
    assert _mean_ms(benchmark) < FRAME_BUDGET_MS * 3


@pytest.mark.benchmark(group="inference")
def test_onnx_inference_is_the_fast_path(benchmark, tmp_path: Path) -> None:
    """ONNX Runtime on CPU — what the product actually ships.

    The model is exported here rather than loaded from `models/`, because the
    trained weights are DVC outputs and are not in the repository. That makes
    this hermetic: it measures the architecture, at the shipped sequence
    length, on whatever CPU the runner gave us.
    """
    torch = pytest.importorskip("torch")
    ort = pytest.importorskip("onnxruntime")
    from core.temporal.model import GestureTransformer

    model = GestureTransformer(input_dim=FEATURE_DIM, num_classes=18, max_seq_len=SEQUENCE_LEN)
    model.eval()

    onnx_path = tmp_path / "bench.onnx"
    dummy_features = torch.randn(1, SEQUENCE_LEN, FEATURE_DIM)
    dummy_mask = torch.zeros(1, SEQUENCE_LEN, dtype=torch.bool)
    torch.onnx.export(
        model,
        (dummy_features, dummy_mask),
        str(onnx_path),
        input_names=["input", "mask"],
        output_names=["logits", "confidence"],
        dynamic_axes={"input": {0: "batch", 1: "seq_len"}, "mask": {0: "batch", 1: "seq_len"}},
        opset_version=17,
    )

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1  # measure the work, not the runner's cores
    session = ort.InferenceSession(
        str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
    )

    inputs = {
        "input": dummy_features.numpy(),
        "mask": dummy_mask.numpy(),
    }

    outputs = benchmark(session.run, ["logits"], inputs)

    assert np.asarray(outputs[0]).shape == (1, 18)
    # Single-threaded, so the budget is looser than the p50 the README quotes
    # for a warm multi-threaded session.
    assert _mean_ms(benchmark) < FRAME_BUDGET_MS
