"""
Smoke test: run the full pipeline in offline modes and assert invariants.
Run:  python test_pipeline.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from roadpipe.config import PipelineConfig
from roadpipe.cli import run


def check_report(rep, label):
    g = rep["graph"]
    assert g["nodes"] > 0, f"{label}: no nodes"
    assert g["edges"] > 0, f"{label}: no edges"
    steps = rep["stress_test"]
    assert steps[0]["resilience_index"] == 1.0, f"{label}: baseline R != 1"
    final = rep["resilience_index_final"]
    assert 0.0 <= final <= 1.0, f"{label}: R out of range ({final})"
    assert len(rep["gatekeepers"]) >= 1, f"{label}: no gatekeepers"
    print(f"  [{label}] nodes={g['nodes']} edges={g['edges']} "
          f"bridges={rep['healing']['bridges_added']} "
          f"R_final={final}  OK")


def main():
    print("Running offline smoke tests...")

    # 1. Image mode on road.png
    img = "road.png"
    if os.path.exists(img):
        cfg = PipelineConfig(mode="image", out_dir="_test_image", verbose=False)
        check_report(run(cfg, input_path=img), "image/road.png")
    else:
        print("  [image] road.png not found, skipping")

    # 2. Synthetic mode
    cfg = PipelineConfig(mode="synth", out_dir="_test_synth", verbose=False)
    check_report(run(cfg), "synth")

    print("All smoke tests passed.")


# pytest entry point.
#
# `main()` above is the documented script interface ("Run: python
# test_pipeline.py") and it works. But pytest only collects callables named
# test_*, so this file contributed zero tests to a `pytest` run: CI installed
# pytest, collected nothing, and reported success without ever executing the
# smoke test. This wrapper makes the existing test visible to pytest. It adds
# no new assertions -- every check still lives in check_report().
def test_pipeline_smoke():
    main()


if __name__ == "__main__":
    main()
