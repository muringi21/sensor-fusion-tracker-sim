"""Regenerate deterministic regression fixtures in tests/fixtures/.

    python scripts/gen_fixtures.py            # overwrite tests/fixtures/
    python scripts/gen_fixtures.py --out DIR  # write elsewhere (used by pytest)

Each fixture contains the scenario parameters, ground truth and the exact
measurement sets. Rust regression tests (tests/regression.rs) load these files,
so `cargo test` needs no Python and the committed files are the single source
of truth. Regenerating must reproduce them: byte-identical on the same CPU
architecture, and within 1e-9 elsewhere (NumPy on ARM vs x86 can differ in the
last bit of some floats). Checked by tests/test_smoke.py.
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
from tracker.multi_sim import DEFAULT_TARGETS, simulate_multi  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "tests" / "fixtures"

SCENARIOS = {
    # Original README demo: two well-separated targets, no clutter.
    "two_targets_clean": dict(seed=0, n=250, clutter_per_step=0, targets=DEFAULT_TARGETS),
    # Same geometry with 5 uniform false alarms per step over a 400x400 area.
    "two_targets_clutter": dict(seed=1, n=250, clutter_per_step=5, targets=DEFAULT_TARGETS),
    # Three targets whose paths cross near the origin around t = 5 s.
    "three_targets_crossing": dict(
        seed=7, n=200, clutter_per_step=1,
        targets=((-50.0, 0.0, 10.0, 0.0), (0.0, -50.0, 0.0, 10.0), (50.0, 50.0, -10.0, -10.0)),
    ),
}

DT, POS_STD, AREA = 0.1, 2.0, 200.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    out = ap.parse_args(argv).out
    out.mkdir(parents=True, exist_ok=True)
    for name, p in SCENARIOS.items():
        truth, meas = simulate_multi(seed=p["seed"], n=p["n"], dt=DT, pos_std=POS_STD,
                                     clutter_per_step=p["clutter_per_step"], area=AREA,
                                     targets=p["targets"])
        doc = {
            "name": name,
            "seed": p["seed"], "steps": p["n"], "dt": DT, "pos_std": POS_STD,
            "clutter_per_step": p["clutter_per_step"], "area": AREA,
            "truth": [[[round(v, 12) for v in s] for s in tgt] for tgt in truth.tolist()],
            "measurements": [[list(m) for m in step] for step in meas],
        }
        path = out / f"{name}.json"
        path.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
