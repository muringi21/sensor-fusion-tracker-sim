"""Seeded multi-target simulation (the "Python experiment layer").

The simulator is pure NumPy so fixtures can be generated without building the
Rust extension. ``run_multi_demo`` feeds the simulated detections to the Rust
``tracker_core.MultiTrackerCV``.
"""

import numpy as np

DEFAULT_TARGETS = (
    (-50.0, -20.0, 12.0, 3.5),
    (60.0, 30.0, -10.0, -2.0),
)


def simulate_multi(seed=0, n=200, dt=0.1, pos_std=2.0, clutter_per_step=0,
                   area=200.0, targets=DEFAULT_TARGETS):
    """Simulate constant-velocity targets with noisy, unlabelled detections.

    Returns ``(truth, measurements)`` where ``truth`` has shape
    ``(n_targets, n, 4)`` holding ``[x, y, vx, vy]`` and ``measurements[k]`` is a
    shuffled list of ``(x, y)`` tuples (detections + uniform clutter).
    Determinism: NumPy's ``default_rng`` (PCG64) gives identical streams for a
    given seed across platforms and NumPy versions >= 1.17.
    """
    rng = np.random.default_rng(seed)
    x0 = np.asarray(targets, dtype=float)
    truth = np.zeros((len(x0), n, 4))
    truth[:, 0] = x0
    for i in range(1, n):
        truth[:, i, 0:2] = truth[:, i - 1, 0:2] + truth[:, i - 1, 2:4] * dt
        truth[:, i, 2:4] = truth[:, i - 1, 2:4]

    measurements = []
    for i in range(n):
        m = []
        for t in range(len(x0)):
            m.append((float(truth[t, i, 0] + rng.normal(0, pos_std)),
                      float(truth[t, i, 1] + rng.normal(0, pos_std))))
        for _ in range(clutter_per_step):
            m.append((float(rng.uniform(-area, area)), float(rng.uniform(-area, area))))
        rng.shuffle(m)
        measurements.append(m)
    return truth, measurements


def run_multi_demo(seed=0, n=250, dt=0.1, clutter_per_step=0):
    """Run the Rust tracker on a simulated scenario.

    Returns ``(X1, X2, measurements, hist)``; ``hist`` maps track id to a list
    of estimated ``(x, y)`` positions.
    """
    import tracker_core

    truth, measurements = simulate_multi(seed=seed, n=n, dt=dt,
                                         clutter_per_step=clutter_per_step)
    mt = tracker_core.MultiTrackerCV(dt=dt, q=1.0, r_pos=4.0, gate_d2=9.21, max_missed=6)
    hist = {}
    for k in range(n):
        for tid, s in mt.step(measurements[k]):
            hist.setdefault(tid, []).append((s[0], s[1]))
    return truth[0], truth[1], measurements, hist
