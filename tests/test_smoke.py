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


FIXTURE_ABS_TOL = 1e-9  # metres; see scripts/gen_fixtures.py


def _assert_close(a, b, path="$"):
    """Exact match for structure/ints/strings, FIXTURE_ABS_TOL for floats."""
    if isinstance(a, float) or isinstance(b, float):
        assert abs(a - b) <= FIXTURE_ABS_TOL, f"{path}: {a!r} vs {b!r}"
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), f"{path}: length differs"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_close(x, y, f"{path}[{i}]")
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), f"{path}: keys differ"
        for k in a:
            _assert_close(a[k], b[k], f"{path}.{k}")
    else:
        assert a == b, f"{path}: {a!r} vs {b!r}"


def test_fixtures_are_reproducible(tmp_path):
    """Regenerating fixtures (into a temp dir, never over the committed files)
    reproduces the committed ones to within FIXTURE_ABS_TOL."""
    subprocess.run([sys.executable, str(ROOT / "scripts/gen_fixtures.py"), "--out", str(tmp_path)],
                   check=True, capture_output=True)
    committed = sorted((ROOT / "tests/fixtures").glob("*.json"))
    assert committed and sorted(p.name for p in tmp_path.glob("*.json")) == [p.name for p in committed]
    for p in committed:
        _assert_close(json.loads((tmp_path / p.name).read_text()), json.loads(p.read_text()), p.name)


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
