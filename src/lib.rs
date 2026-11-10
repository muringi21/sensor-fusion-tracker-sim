//! `tracker_core`: a small multi-target tracking library.
//!
//! * [`kalman`] — 2D constant-velocity Kalman filter.
//! * [`tracker`] — gating, greedy nearest-neighbour association, track lifecycle.
//! * [`scenario`] — deterministic synthetic scenarios used by tests and benchmarks.
//!
//! Python bindings (PyO3) are compiled only with the `python` feature.

pub mod kalman;
pub mod scenario;
pub mod tracker;

pub use kalman::KalmanFilterCV;
pub use tracker::{MultiTracker, Track, TrackerConfig};

#[cfg(feature = "python")]
mod python;
