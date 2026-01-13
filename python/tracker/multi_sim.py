import numpy as np
import tracker_core

def simulate_multi(seed=0, n=200, dt=0.1, pos_std=2.0, clutter_per_step=0, area=200.0):
    rng = np.random.default_rng(seed)

    # Two targets
    x1 = np.array([-50.0, -20.0, 12.0,  3.5])
    x2 = np.array([ 60.0,  30.0,-10.0, -2.0])

    X1 = np.zeros((n, 4))
    X2 = np.zeros((n, 4))
    X1[0] = x1
    X2[0] = x2

    for i in range(1, n):
        for X in (X1, X2):
            X[i, 0] = X[i - 1, 0] + X[i - 1, 2] * dt
            X[i, 1] = X[i - 1, 1] + X[i - 1, 3] * dt
            X[i, 2] = X[i - 1, 2]
            X[i, 3] = X[i - 1, 3]

    # Per step: measurement set (unordered), may include clutter
    measurements = []
    for i in range(n):
        m = []

        # detections (no IDs)
        for X in (X1, X2):
            mx = X[i, 0] + rng.normal(0, pos_std)
            my = X[i, 1] + rng.normal(0, pos_std)
            m.append((float(mx), float(my)))

        # clutter (optional)
        for _ in range(clutter_per_step):
            m.append((float(rng.uniform(-area, area)), float(rng.uniform(-area, area))))

        rng.shuffle(m)
        measurements.append(m)

    return X1, X2, measurements

def run_multi_demo(seed=0, n=250, dt=0.1, clutter_per_step=0):
    X1, X2, measurements = simulate_multi(
        seed=seed, n=n, dt=dt,
        clutter_per_step=clutter_per_step
    )

    mt = tracker_core.MultiTrackerCV(dt=dt, q=1.0, r_pos=4.0, gate_d2=9.21, max_missed=6)

    # track_id -> list[(x,y)]
    hist = {}
    for k in range(n):
        states = mt.step(measurements[k])  # [(id, [x,y,vx,vy]), ...]
        for tid, s in states:
            hist.setdefault(tid, []).append((s[0], s[1]))

    return X1, X2, measurements, hist