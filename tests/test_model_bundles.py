"""Tests for the trained model bundles this repo ships.

Everything else under tests/ exercises code. This file exercises the two
artifacts in `models/`, because a clone with no weights is a clone that cannot
run the product, and nothing else would notice: the web build only copies
`public/`, so it stays green whether or not a model is there.

The bundle's `labels.json` states the vocabulary, window and feature width the
graph was exported with. Those three can drift from the graph, and from the
evaluation reports quoted in the README, without any import failing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from core.inference.onnx_runtime import ONNXInferenceRuntime

REPO_ROOT = Path(__file__).resolve().parent.parent

# Bundle -> the report holding the accuracy its manifest claims.
SHIPPED_BUNDLES = {
    "hagrid": "eval_hagrid_full.json",
    "asl_alphabet": "eval_asl_alphabet.json",
}


def _manifest(name: str) -> dict:
    return json.loads((REPO_ROOT / "models" / name / "labels.json").read_text())


@pytest.mark.parametrize("name", sorted(SHIPPED_BUNDLES))
class TestShippedBundle:
    def test_weights_are_present(self, name: str) -> None:
        """The graph and its external weights are both in the tree.

        `.gitignore` excludes `*.onnx` wholesale and re-admits these two by
        name, so a rename or a new bundle silently falls back to ignored.
        """
        bundle = REPO_ROOT / "models" / name
        graph = bundle / "gesture.onnx"
        assert graph.is_file(), f"{graph} missing — was the bundle committed?"

        # The exporter splits weights out past a size threshold. If it did, the
        # graph alone is a few tens of KB of metadata and loads into nothing.
        weights = bundle / "gesture.onnx.data"
        if graph.stat().st_size < 1_000_000:
            assert weights.is_file(), (
                f"{graph.name} is too small to hold weights and {weights.name} "
                "is absent; the bundle would load as an empty graph"
            )

    def test_graph_agrees_with_its_manifest(self, name: str) -> None:
        """Class count and feature width match what labels.json advertises.

        The web engine sizes its input ring buffer from the manifest and reads
        logits positionally against the label list, so a mismatch here is a
        wrong label on every prediction rather than an error.
        """
        manifest = _manifest(name)
        runtime = ONNXInferenceRuntime()
        runtime.load(REPO_ROOT / "models" / name / "gesture.onnx")
        try:
            seq_len, feature_dim = manifest["seq_len"], manifest["feature_dim"]
            out = runtime.predict(
                {
                    "input": np.zeros((1, seq_len, feature_dim), dtype=np.float32),
                    "mask": np.zeros((1, seq_len), dtype=bool),
                }
            )
            assert out["logits"].shape == (1, len(manifest["labels"]))
        finally:
            runtime.close()

    def test_manifest_accuracy_matches_its_report(self, name: str) -> None:
        """The number in the bundle is the number the eval actually produced.

        The README quotes these, so a bundle retrained without re-running eval
        would leave the published accuracy describing a model that no longer
        exists.
        """
        report = json.loads((REPO_ROOT / "reports" / SHIPPED_BUNDLES[name]).read_text())
        assert _manifest(name)["test_accuracy"] == pytest.approx(report["accuracy"], abs=5e-5)


def test_dashboard_and_sync_script_offer_the_same_bundles() -> None:
    """The two hardcoded bundle lists stay in step.

    `sync-runtime.sh` decides which bundles get copied into `public/`; the
    dashboard decides which ones it offers. A bundle in the second list but not
    the first is a 404 at model load, which surfaces to the user as the camera
    running with nothing ever recognised.
    """
    script = (REPO_ROOT / "apps/web/scripts/sync-runtime.sh").read_text()
    synced = re.search(r"^BUNDLES=\(([^)]*)\)", script, re.MULTILINE)
    assert synced, "BUNDLES array not found in sync-runtime.sh"

    page = (REPO_ROOT / "apps/web/src/app/dashboard/page.tsx").read_text()
    offered = re.findall(r'\{\s*id:\s*"([^"]+)",\s*name:', page)

    assert set(synced.group(1).split()) == set(offered) == set(SHIPPED_BUNDLES)
