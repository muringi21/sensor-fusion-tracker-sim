"""Python-side tests: bindings, fixture determinism, end-to-end parity."""

import json
import pathlib
import subprocess
import sys

import tracker_core
from tracker.multi_sim import simulate_multi, run_multi_demo

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_import_and_step():
    mt = tracker_core.MultiTrackerCV(dt=0.1)
    out = mt.step([(0.0, 0.0), (10.0, 0.0)])
    assert [tid for tid, _ in out] == [1, 2]
    assert all(len(s) == 4 for _, s in out)


def test_kalman_binding_predict():
    kf = tracker_core.KalmanFilterCV(0.1, x0=[1.0, 2.0, 3.0, -4.0])
    kf.predict()
    x, y, vx, vy = kf.state()
    assert abs(x - 1.3) < 1e-12 and abs(y - 1.6) < 1e-12


def test_simulation_is_seeded():
    a = simulate_multi(seed=5, n=20, clutter_per_step=3)[1]
    b = simulate_multi(seed=5, n=20, clutter_per_step=3)[1]
    c = simulate_multi(seed=6, n=20, clutter_per_step=3)[1]
    assert a == b and a != c


def test_fixtures_are_reproducible(tmp_path):
    """Committed fixtures must equal a fresh regeneration byte-for-byte."""
    before = {p.name: p.read_bytes() for p in (ROOT / "tests/fixtures").glob("*.json")}
    subprocess.run([sys.executable, str(ROOT / "scripts/gen_fixtures.py")], check=True)
    after = {p.name: p.read_bytes() for p in (ROOT / "tests/fixtures").glob("*.json")}
    assert before == after


def test_python_path_matches_rust_golden():
    """Python -> PyO3 -> Rust gives the same final tracks as the Rust golden."""
    fx = json.loads((ROOT / "tests/fixtures/two_targets_clean.json").read_text())
    gold = json.loads((ROOT / "tests/golden/two_targets_clean.json").read_text())
    mt = tracker_core.MultiTrackerCV(dt=fx["dt"], max_missed=6)
    for m in fx["measurements"]:
        out = mt.step([tuple(p) for p in m])
    assert [tid for tid, _ in out] == [tid for tid, _ in gold["final_tracks"]]
    for (_, s), (_, g) in zip(out, gold["final_tracks"]):
        assert max(abs(a - b) for a, b in zip(s, g)) < 1e-9


def test_demo_runs():
    X1, X2, meas, hist = run_multi_demo(seed=0, n=50)
    assert len(meas) == 50 and hist
