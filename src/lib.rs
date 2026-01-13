use nalgebra::{Matrix2, Matrix2x4, Matrix4, Vector2, Vector4};
use pyo3::prelude::*;

fn f_cv(dt: f64) -> Matrix4<f64> {
    Matrix4::new(
        1.0, 0.0, dt,  0.0,
        0.0, 1.0, 0.0, dt,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
}

fn q_cv(dt: f64, q: f64) -> Matrix4<f64> {
    let dt2 = dt * dt;
    let dt3 = dt2 * dt;
    let dt4 = dt2 * dt2;

    Matrix4::new(
        dt4 / 4.0, 0.0,       dt3 / 2.0, 0.0,
        0.0,       dt4 / 4.0, 0.0,       dt3 / 2.0,
        dt3 / 2.0, 0.0,       dt2,       0.0,
        0.0,       dt3 / 2.0, 0.0,       dt2,
    ) * q
}

fn h_pos() -> Matrix2x4<f64> {
    Matrix2x4::new(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
    )
}

fn mahalanobis2(y: &Vector2<f64>, s: &Matrix2<f64>) -> Option<f64> {
    let s_inv = s.try_inverse()?;
    Some((y.transpose() * s_inv * y)[0])
}

#[pyclass]
#[derive(Clone)]
pub struct KalmanFilterCV {
    dt: f64,
    q: f64,
    r_pos: f64,
    x: Vector4<f64>,
    p: Matrix4<f64>,
}

#[pymethods]
impl KalmanFilterCV {
    #[new]
    #[pyo3(signature = (dt, q=1.0, r_pos=1.0, x0=None))]
    fn new(dt: f64, q: f64, r_pos: f64, x0: Option<Vec<f64>>) -> PyResult<Self> {
        let x = if let Some(v) = x0 {
            if v.len() != 4 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "x0 must have length 4: [x, y, vx, vy]",
                ));
            }
            Vector4::new(v[0], v[1], v[2], v[3])
        } else {
            Vector4::zeros()
        };

        let p = Matrix4::identity() * 1000.0;
        Ok(Self { dt, q, r_pos, x, p })
    }

    fn state(&self) -> Vec<f64> {
        vec![self.x[0], self.x[1], self.x[2], self.x[3]]
    }

    fn predict(&mut self) {
        let f = f_cv(self.dt);
        let q = q_cv(self.dt, self.q);
        self.x = f * self.x;
        self.p = f * self.p * f.transpose() + q;
    }

    fn update_pos(&mut self, zx: f64, zy: f64) -> PyResult<Vec<f64>> {
        let h = h_pos();
        let z = Vector2::new(zx, zy);
        let r = Matrix2::identity() * self.r_pos;

        let z_pred = h * self.x;
        let y = z - z_pred;
        let s = h * self.p * h.transpose() + r;

        let s_inv = s
            .try_inverse()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("S not invertible"))?;
        let k = self.p * h.transpose() * s_inv;

        self.x = self.x + k * y;
        self.p = (Matrix4::identity() - k * h) * self.p;
        Ok(self.state())
    }
}

#[derive(Clone)]
struct Track {
    id: u64,
    kf: KalmanFilterCV,
    missed: u32,
}

#[pyclass]
pub struct MultiTrackerCV {
    dt: f64,
    q: f64,
    r_pos: f64,
    gate_d2: f64,
    max_missed: u32,
    next_id: u64,
    tracks: Vec<Track>,
}

#[pymethods]
impl MultiTrackerCV {
    #[new]
    #[pyo3(signature = (dt, q=1.0, r_pos=4.0, gate_d2=9.21, max_missed=5))]
    fn new(dt: f64, q: f64, r_pos: f64, gate_d2: f64, max_missed: u32) -> Self {
        Self { dt, q, r_pos, gate_d2, max_missed, next_id: 1, tracks: Vec::new() }
    }

    /// measurements: list of (x,y)
    /// returns: list of (track_id, [x,y,vx,vy])
    fn step(&mut self, measurements: Vec<(f64, f64)>) -> PyResult<Vec<(u64, Vec<f64>)>> {
        // Predict all
        for trk in &mut self.tracks {
            trk.kf.predict();
        }

        let h = h_pos();
        let r = Matrix2::identity() * self.r_pos;

        // Candidate (d2, ti, mi)
        let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
        for (ti, trk) in self.tracks.iter().enumerate() {
            let z_pred = h * trk.kf.x;
            let s = h * trk.kf.p * h.transpose() + r;

            for (mi, (mx, my)) in measurements.iter().enumerate() {
                let y = Vector2::new(*mx, *my) - z_pred;
                if let Some(d2) = mahalanobis2(&y, &s) {
                    if d2 <= self.gate_d2 {
                        pairs.push((d2, ti, mi));
                    }
                }
            }
        }

        // Greedy NN assignment
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut track_used = vec![false; self.tracks.len()];
        let mut meas_used = vec![false; measurements.len()];
        let mut assigned: Vec<(usize, usize)> = Vec::new();

        for (_d2, ti, mi) in pairs {
            if !track_used[ti] && !meas_used[mi] {
                track_used[ti] = true;
                meas_used[mi] = true;
                assigned.push((ti, mi));
            }
        }

        // Update assigned
        for (ti, mi) in assigned {
            let (mx, my) = measurements[mi];
            let _ = self.tracks[ti].kf.update_pos(mx, my)?;
            self.tracks[ti].missed = 0;
        }

        // Missed tracks
        for (ti, trk) in self.tracks.iter_mut().enumerate() {
            if !track_used[ti] {
                trk.missed += 1;
            }
        }

        // Spawn new tracks for unused measurements
        for (mi, (mx, my)) in measurements.iter().enumerate() {
            if !meas_used[mi] {
                let x0 = vec![*mx, *my, 0.0, 0.0];
                let kf = KalmanFilterCV::new(self.dt, self.q, self.r_pos, Some(x0))?;
                let id = self.next_id;
                self.next_id += 1;
                self.tracks.push(Track { id, kf, missed: 0 });
            }
        }

        // Prune stale
        self.tracks.retain(|t| t.missed <= self.max_missed);

        Ok(self.tracks.iter().map(|t| (t.id, t.kf.state())).collect())
    }
}

#[pymodule]
fn tracker_core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<KalmanFilterCV>()?;
    m.add_class::<MultiTrackerCV>()?;
    Ok(())
}