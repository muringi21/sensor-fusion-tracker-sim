# Sensor Fusion Tracker Sim (Rust Core + Python)

A small **multi-target tracking** sandbox that combines a high-performance **Rust** tracking core with a **Python** experiment layer. It simulates multiple moving targets with noisy detections (and optional clutter), then tracks them using **Kalman filtering + data association**. The goal is to demonstrate production-style engineering patterns: a fast, testable systems core with a clean Python-facing API for rapid iteration and visualization.


![demo](assets/demo.png)
---

## Features
- **Rust core (PyO3 + maturin)**
  - 2D constant-velocity Kalman filter per track: state `[x, y, vx, vy]`
  - Track management: predict → associate → update, spawn new tracks, prune stale tracks
  - **Data association**
    - **Mahalanobis gating** to reject unlikely track–measurement pairs
    - **Greedy nearest-neighbor assignment** (lowest Mahalanobis distance first)
- **Python experiment layer**
  - Deterministic multi-target simulation (seeded)
  - Optional clutter generation
  - Quick plotting script to visualize ground truth, measurements, and track outputs

---

## Architecture
- **Rust (`tracker_core`)**
  - Implements the math-heavy tracking logic (KF + gating + association)
  - Exposed to Python as a native extension module via **PyO3**
- **Python (`python/tracker/`)**
  - Simulation and plotting utilities
  - Calls into `tracker_core.MultiTrackerCV` each timestep

---

## Quickstart (macOS, Python 3.12)
Prereqs:
- Python 3.12+
- Rust toolchain (`rustc`, `cargo`) via rustup
- Xcode command line tools

```bash
# 1) Create and activate venv
python3.12 -m venv .venv
source .venv/bin/activate

# 2) Install build + runtime deps
python -m pip install -U pip maturin numpy matplotlib pytest

# 3) Build/install the Rust extension into the venv
maturin develop

# 4) Run the multi-target demo (plot)
python -m tracker.run_multi

# 5) Run tests
pytest -q