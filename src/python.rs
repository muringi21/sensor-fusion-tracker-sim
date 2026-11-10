//! PyO3 bindings. Thin wrappers: all logic lives in the pure-Rust modules.

use crate::kalman::{KalmanError, KalmanFilterCV as CoreKf};
use crate::tracker::{MultiTracker, TrackerConfig};
use nalgebra::Vector4;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

fn to_py(e: KalmanError) -> PyErr {
    PyRuntimeError::new_err(format!("{e:?}"))
}

#[pyclass(name = "KalmanFilterCV")]
#[derive(Clone)]
pub struct PyKalmanFilterCV(CoreKf);

#[pymethods]
impl PyKalmanFilterCV {
    #[new]
    #[pyo3(signature = (dt, q=1.0, r_pos=1.0, x0=None))]
    fn new(dt: f64, q: f64, r_pos: f64, x0: Option<Vec<f64>>) -> PyResult<Self> {
        let x = match x0 {
            Some(v) if v.len() == 4 => Vector4::new(v[0], v[1], v[2], v[3]),
            Some(_) => {
                return Err(PyValueError::new_err(
                    "x0 must have length 4: [x, y, vx, vy]",
                ))
            }
            None => Vector4::zeros(),
        };
        Ok(Self(CoreKf::new(dt, q, r_pos, x)))
    }
    fn state(&self) -> Vec<f64> {
        self.0.state().to_vec()
    }
    fn predict(&mut self) {
        self.0.predict()
    }
    fn update_pos(&mut self, zx: f64, zy: f64) -> PyResult<Vec<f64>> {
        self.0.update_pos(zx, zy).map_err(to_py)?;
        Ok(self.state())
    }
}

#[pyclass(name = "MultiTrackerCV")]
pub struct PyMultiTrackerCV(MultiTracker);

#[pymethods]
impl PyMultiTrackerCV {
    #[new]
    #[pyo3(signature = (dt, q=1.0, r_pos=4.0, gate_d2=9.21, max_missed=5))]
    fn new(dt: f64, q: f64, r_pos: f64, gate_d2: f64, max_missed: u32) -> Self {
        Self(MultiTracker::new(TrackerConfig {
            dt,
            q,
            r_pos,
            gate_d2,
            max_missed,
        }))
    }
    /// measurements: list of (x, y). Returns list of (track_id, [x, y, vx, vy]).
    fn step(&mut self, measurements: Vec<(f64, f64)>) -> PyResult<Vec<(u64, Vec<f64>)>> {
        let tracks = self.0.step(&measurements).map_err(to_py)?;
        Ok(tracks
            .iter()
            .map(|t| (t.id, t.kf.state().to_vec()))
            .collect())
    }
}

#[pymodule]
fn tracker_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKalmanFilterCV>()?;
    m.add_class::<PyMultiTrackerCV>()?;
    Ok(())
}
