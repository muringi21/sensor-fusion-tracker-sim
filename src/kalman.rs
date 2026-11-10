//! 2D constant-velocity (CV) Kalman filter.
//!
//! State `x = [px, py, vx, vy]`, position-only measurements `z = [px, py]`.
//! Process noise is the discrete white-noise-acceleration model with
//! spectral density `q`; measurement noise is isotropic with variance `r_pos`.

use nalgebra::{Matrix2, Matrix2x4, Matrix4, Vector2, Vector4};

/// State transition `F(dt)` for the CV model.
pub fn f_cv(dt: f64) -> Matrix4<f64> {
    #[rustfmt::skip]
    let f = Matrix4::new(
        1.0, 0.0, dt,  0.0,
        0.0, 1.0, 0.0, dt,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    );
    f
}

/// Process noise `Q(dt, q)` (discrete white-noise acceleration).
pub fn q_cv(dt: f64, q: f64) -> Matrix4<f64> {
    let dt2 = dt * dt;
    let dt3 = dt2 * dt;
    let dt4 = dt2 * dt2;
    #[rustfmt::skip]
    let m = Matrix4::new(
        dt4 / 4.0, 0.0,       dt3 / 2.0, 0.0,
        0.0,       dt4 / 4.0, 0.0,       dt3 / 2.0,
        dt3 / 2.0, 0.0,       dt2,       0.0,
        0.0,       dt3 / 2.0, 0.0,       dt2,
    );
    m * q
}

/// Position measurement matrix `H`.
pub fn h_pos() -> Matrix2x4<f64> {
    #[rustfmt::skip]
    let h = Matrix2x4::new(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
    );
    h
}

/// Initial covariance used for newly spawned tracks (diffuse prior).
pub const P0_SCALE: f64 = 1000.0;

#[derive(Debug, Clone, PartialEq)]
pub enum KalmanError {
    SingularInnovation,
}

#[derive(Debug, Clone)]
pub struct KalmanFilterCV {
    pub dt: f64,
    pub q: f64,
    pub r_pos: f64,
    pub x: Vector4<f64>,
    pub p: Matrix4<f64>,
    f: Matrix4<f64>,
    qm: Matrix4<f64>,
}

impl KalmanFilterCV {
    pub fn new(dt: f64, q: f64, r_pos: f64, x0: Vector4<f64>) -> Self {
        Self {
            dt,
            q,
            r_pos,
            x: x0,
            p: Matrix4::identity() * P0_SCALE,
            f: f_cv(dt),
            qm: q_cv(dt, q),
        }
    }

    pub fn state(&self) -> [f64; 4] {
        [self.x[0], self.x[1], self.x[2], self.x[3]]
    }

    /// Time update: `x <- F x`, `P <- F P F^T + Q`.
    pub fn predict(&mut self) {
        self.x = self.f * self.x;
        self.p = self.f * self.p * self.f.transpose() + self.qm;
    }

    /// Predicted measurement `H x` and innovation covariance `S = H P H^T + R`.
    pub fn innovation(&self) -> (Vector2<f64>, Matrix2<f64>) {
        let h = h_pos();
        let s = h * self.p * h.transpose() + Matrix2::identity() * self.r_pos;
        (h * self.x, s)
    }

    /// Measurement update with a position measurement.
    ///
    /// Uses the Joseph form `P = (I-KH) P (I-KH)^T + K R K^T`, which keeps `P`
    /// symmetric positive semi-definite under rounding.
    pub fn update_pos(&mut self, zx: f64, zy: f64) -> Result<(), KalmanError> {
        let h = h_pos();
        let r = Matrix2::identity() * self.r_pos;
        let (z_pred, s) = self.innovation();
        let y = Vector2::new(zx, zy) - z_pred;
        let s_inv = s.try_inverse().ok_or(KalmanError::SingularInnovation)?;
        let k = self.p * h.transpose() * s_inv;
        self.x += k * y;
        let ikh = Matrix4::identity() - k * h;
        self.p = ikh * self.p * ikh.transpose() + k * r * k.transpose();
        Ok(())
    }
}

/// Squared Mahalanobis distance `y^T S^-1 y`. Returns `None` if `S` is singular.
pub fn mahalanobis2(y: &Vector2<f64>, s: &Matrix2<f64>) -> Option<f64> {
    let s_inv = s.try_inverse()?;
    Some((y.transpose() * s_inv * y)[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    fn kf(x0: [f64; 4]) -> KalmanFilterCV {
        KalmanFilterCV::new(0.1, 1.0, 4.0, Vector4::from(x0))
    }

    #[test]
    fn predict_propagates_constant_velocity_state() {
        let mut f = kf([1.0, 2.0, 3.0, -4.0]);
        f.predict();
        let s = f.state();
        assert!((s[0] - 1.3).abs() < EPS);
        assert!((s[1] - 1.6).abs() < EPS);
        assert_eq!(s[2], 3.0);
        assert_eq!(s[3], -4.0);
    }

    #[test]
    fn n_predicts_equal_one_predict_of_n_dt() {
        let mut a = kf([0.0, 0.0, 2.0, 1.0]);
        for _ in 0..10 {
            a.predict();
        }
        let s = a.state();
        assert!((s[0] - 2.0).abs() < 1e-12 && (s[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn predict_matches_closed_form_covariance() {
        // From P0 = 0 a single predict must give exactly Q.
        let mut f = kf([0.0; 4]);
        f.p = Matrix4::zeros();
        f.predict();
        assert!((f.p - q_cv(0.1, 1.0)).abs().max() < EPS);
    }

    #[test]
    fn q_is_symmetric_and_scales_linearly() {
        let q1 = q_cv(0.25, 1.0);
        assert!((q1 - q1.transpose()).abs().max() < EPS);
        assert!((q_cv(0.25, 3.0) - q1 * 3.0).abs().max() < EPS);
        assert!(q1.symmetric_eigenvalues().iter().all(|&e| e > -EPS));
    }

    #[test]
    fn predict_grows_uncertainty() {
        let mut f = kf([0.0; 4]);
        let tr0 = f.p.trace();
        f.predict();
        assert!(f.p.trace() > tr0);
    }

    #[test]
    fn update_pulls_state_toward_measurement_and_shrinks_covariance() {
        let mut f = kf([0.0; 4]);
        f.predict();
        let tr_before = f.p.fixed_view::<2, 2>(0, 0).trace();
        f.update_pos(10.0, -10.0).unwrap();
        let s = f.state();
        // Diffuse prior (P0=1000) vs R=4: gain close to 1 → estimate near z.
        assert!(s[0] > 9.9 && s[0] < 10.0, "{s:?}");
        assert!(s[1] < -9.9 && s[1] > -10.0, "{s:?}");
        assert!(f.p.fixed_view::<2, 2>(0, 0).trace() < tr_before);
    }

    #[test]
    fn scalar_case_matches_hand_computed_gain() {
        // With P = diag(p), H selects position, gain on x is p/(p+r).
        let mut f = kf([0.0; 4]);
        f.p = Matrix4::identity() * 6.0;
        f.update_pos(5.0, 0.0).unwrap();
        let k = 6.0 / (6.0 + 4.0);
        assert!((f.state()[0] - k * 5.0).abs() < EPS);
        assert!((f.p[(0, 0)] - (1.0 - k) * 6.0).abs() < 1e-12);
    }

    #[test]
    fn covariance_stays_symmetric_psd_over_long_run() {
        let mut f = kf([0.0; 4]);
        for i in 0..10_000 {
            f.predict();
            let t = i as f64 * 0.1;
            f.update_pos(t, -t).unwrap();
        }
        assert!((f.p - f.p.transpose()).abs().max() < 1e-9);
        assert!(f.p.symmetric_eigenvalues().iter().all(|&e| e > 0.0));
    }

    #[test]
    fn converges_on_noise_free_constant_velocity_target() {
        let mut f = kf([0.0; 4]);
        let (vx, vy) = (3.0, -1.5);
        for k in 1..=300 {
            f.predict();
            let t = k as f64 * 0.1;
            f.update_pos(vx * t, vy * t).unwrap();
        }
        let s = f.state();
        assert!(
            (s[2] - vx).abs() < 1e-3 && (s[3] - vy).abs() < 1e-3,
            "{s:?}"
        );
    }

    #[test]
    fn mahalanobis_identity_is_euclidean() {
        let y = Vector2::new(3.0, 4.0);
        assert!((mahalanobis2(&y, &Matrix2::identity()).unwrap() - 25.0).abs() < EPS);
        assert!(mahalanobis2(&y, &Matrix2::zeros()).is_none());
    }

    #[test]
    fn singular_innovation_is_reported() {
        let mut f = KalmanFilterCV::new(0.1, 1.0, 0.0, Vector4::zeros());
        f.p = Matrix4::zeros();
        assert_eq!(f.update_pos(1.0, 1.0), Err(KalmanError::SingularInnovation));
    }
}
