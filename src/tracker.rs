//! Multi-target tracker: predict → gate → greedy NN associate → update →
//! spawn → prune.

use crate::kalman::{KalmanError, KalmanFilterCV};
use nalgebra::Vector4;

#[derive(Debug, Clone, Copy)]
pub struct TrackerConfig {
    pub dt: f64,
    /// Process-noise spectral density.
    pub q: f64,
    /// Position measurement variance.
    pub r_pos: f64,
    /// Gate on squared Mahalanobis distance (9.21 = chi^2, 2 dof, 99%).
    pub gate_d2: f64,
    /// A track is deleted once it misses more than this many consecutive steps.
    pub max_missed: u32,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            dt: 0.1,
            q: 1.0,
            r_pos: 4.0,
            gate_d2: 9.21,
            max_missed: 5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Track {
    pub id: u64,
    pub kf: KalmanFilterCV,
    pub missed: u32,
    /// Number of measurement updates received (for confirmation logic downstream).
    pub hits: u32,
}

#[derive(Debug, Clone)]
pub struct MultiTracker {
    pub cfg: TrackerConfig,
    next_id: u64,
    tracks: Vec<Track>,
    // Scratch buffers reused across steps.
    pairs: Vec<(f64, usize, usize)>,
    track_used: Vec<bool>,
    meas_used: Vec<bool>,
    order: Vec<usize>,
    sorted_x: Vec<f64>,
}

impl MultiTracker {
    pub fn new(cfg: TrackerConfig) -> Self {
        Self {
            cfg,
            next_id: 1,
            tracks: Vec::new(),
            pairs: Vec::new(),
            track_used: Vec::new(),
            meas_used: Vec::new(),
            order: Vec::new(),
            sorted_x: Vec::new(),
        }
    }

    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    /// Advance one time step with an unordered set of `(x, y)` detections.
    pub fn step(&mut self, measurements: &[(f64, f64)]) -> Result<&[Track], KalmanError> {
        for trk in &mut self.tracks {
            trk.kf.predict();
        }

        // Gating: candidate (cost, track, measurement) pairs.
        //
        // Performance (see README "Performance"): the original loop tested
        // every track against every measurement, O(T*M). Profiling showed ~80%
        // of runtime there. Instead we sort measurements by x once per step and,
        // for each track, binary-search the x-window that can possibly pass the
        // gate. The window is exact, not a heuristic: for an ellipse
        // y^T S^-1 y <= g the extreme x-offset is sqrt(g * S_xx) (likewise y),
        // so no gated pair is ever skipped. Pairs are then fully sorted with
        // index tie-breakers, so the output is identical to the brute-force
        // version (enforced by the golden regression tests).
        self.pairs.clear();
        let gate = self.cfg.gate_d2;
        self.order.clear();
        self.order.extend(0..measurements.len());
        self.order.sort_unstable_by(|&i, &j| {
            measurements[i]
                .0
                .total_cmp(&measurements[j].0)
                .then(i.cmp(&j))
        });
        self.sorted_x.clear();
        self.sorted_x
            .extend(self.order.iter().map(|&i| measurements[i].0));

        for (ti, trk) in self.tracks.iter().enumerate() {
            let (z_pred, s) = trk.kf.innovation();
            let det = s[(0, 0)] * s[(1, 1)] - s[(0, 1)] * s[(1, 0)];
            if !(det.is_finite() && det > 0.0) {
                continue; // singular / non-PD innovation: track cannot gate
            }
            let inv_det = 1.0 / det;
            let (a, b, c) = (
                s[(1, 1)] * inv_det,
                -s[(0, 1)] * inv_det,
                s[(0, 0)] * inv_det,
            );
            let ln_det = det.ln();
            let (px, py) = (z_pred[0], z_pred[1]);
            // Exact bounding box of the gate ellipse (tiny epsilon for rounding).
            let rx = (gate * s[(0, 0)]).sqrt() * (1.0 + 1e-9);
            let ry = (gate * s[(1, 1)]).sqrt() * (1.0 + 1e-9);
            let lo = self.sorted_x.partition_point(|&x| x < px - rx);
            for &mi in &self.order[lo..] {
                let (mx, my) = measurements[mi];
                let dx = mx - px;
                if dx > rx {
                    break;
                }
                let dy = my - py;
                if dy.abs() > ry {
                    continue;
                }
                let d2 = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy;
                if d2 <= gate {
                    // Rank by negative log-likelihood (d2 + ln|S|), not raw
                    // d2: raw d2 favours young, high-covariance tracks, which
                    // then steal detections from established tracks.
                    self.pairs.push((d2 + ln_det, ti, mi));
                }
            }
        }

        // Greedy nearest-neighbour: lowest d2 first. Ties broken by indices so
        // the result is fully deterministic.
        self.pairs
            .sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        self.track_used.clear();
        self.track_used.resize(self.tracks.len(), false);
        self.meas_used.clear();
        self.meas_used.resize(measurements.len(), false);

        for &(_, ti, mi) in &self.pairs {
            if !self.track_used[ti] && !self.meas_used[mi] {
                self.track_used[ti] = true;
                self.meas_used[mi] = true;
                let (mx, my) = measurements[mi];
                let trk = &mut self.tracks[ti];
                trk.kf.update_pos(mx, my)?;
                trk.missed = 0;
                trk.hits += 1;
            }
        }

        for (ti, trk) in self.tracks.iter_mut().enumerate() {
            if !self.track_used[ti] {
                trk.missed += 1;
            }
        }

        // Spawn tracks for unassociated measurements.
        let c = self.cfg;
        for (mi, &(mx, my)) in measurements.iter().enumerate() {
            if !self.meas_used[mi] {
                let kf = KalmanFilterCV::new(c.dt, c.q, c.r_pos, Vector4::new(mx, my, 0.0, 0.0));
                self.tracks.push(Track {
                    id: self.next_id,
                    kf,
                    missed: 0,
                    hits: 1,
                });
                self.next_id += 1;
            }
        }

        self.tracks.retain(|t| t.missed <= c.max_missed);
        Ok(&self.tracks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tracker() -> MultiTracker {
        MultiTracker::new(TrackerConfig::default())
    }

    #[test]
    fn spawns_one_track_per_unassociated_measurement() {
        let mut t = tracker();
        let out = t.step(&[(0.0, 0.0), (100.0, 0.0)]).unwrap();
        assert_eq!(out.iter().map(|t| t.id).collect::<Vec<_>>(), vec![1, 2]);
    }

    #[test]
    fn keeps_identity_for_consistent_detections() {
        let mut t = tracker();
        for k in 0..50 {
            let x = k as f64 * 0.5;
            // Present measurements in alternating order: association must not care.
            let m = if k % 2 == 0 {
                [(x, 0.0), (x, 100.0)]
            } else {
                [(x, 100.0), (x, 0.0)]
            };
            t.step(&m).unwrap();
        }
        let ids: Vec<u64> = t.tracks().iter().map(|t| t.id).collect();
        assert_eq!(ids, vec![1, 2]);
        let low = &t.tracks()[0];
        assert!(low.kf.x[1].abs() < 1.0);
        assert!((low.kf.x[2] - 5.0).abs() < 0.5, "vx={}", low.kf.x[2]);
    }

    #[test]
    fn measurement_outside_gate_spawns_new_track() {
        let mut t = tracker();
        for _ in 0..20 {
            t.step(&[(0.0, 0.0)]).unwrap();
        }
        let out = t.step(&[(50.0, 50.0)]).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[1].id, 2);
    }

    #[test]
    fn prunes_after_max_missed_steps() {
        let mut t = tracker();
        t.step(&[(0.0, 0.0)]).unwrap();
        for _ in 0..5 {
            assert_eq!(t.step(&[]).unwrap().len(), 1);
        }
        assert_eq!(t.step(&[]).unwrap().len(), 0);
    }

    #[test]
    fn greedy_assigns_closest_pair_first() {
        let mut t = tracker();
        for _ in 0..30 {
            t.step(&[(0.0, 0.0), (6.0, 0.0)]).unwrap();
        }
        // A detection at 1.0 must go to track 1 (closest); 5.5 to track 2.
        t.step(&[(5.5, 0.0), (1.0, 0.0)]).unwrap();
        let tr = t.tracks();
        assert_eq!(tr.len(), 2);
        assert!(tr[0].kf.x[0] < 3.0 && tr[1].kf.x[0] > 3.0);
    }

    #[test]
    fn young_track_does_not_steal_from_established_track() {
        // Established track at origin; a fresh (diffuse-covariance) track is
        // spawned 8 m away. A detection at 1 m belongs to the established
        // track even though its raw Mahalanobis d2 to the young track is smaller.
        let mut t = tracker();
        for _ in 0..30 {
            t.step(&[(0.0, 0.0)]).unwrap();
        }
        t.step(&[(0.0, 0.0), (8.0, 0.0)]).unwrap();
        t.step(&[(1.0, 0.0)]).unwrap();
        let old = &t.tracks()[0];
        assert_eq!(old.id, 1);
        assert_eq!(old.missed, 0, "established track lost its detection");
    }

    #[test]
    fn closed_form_d2_matches_reference_mahalanobis() {
        use crate::kalman::mahalanobis2;
        use nalgebra::{Matrix2, Vector2};
        let s = Matrix2::new(5.0, 1.2, 1.2, 3.0);
        let inv = s.try_inverse().unwrap();
        for &(dx, dy) in &[(0.0, 0.0), (1.0, -2.0), (3.5, 0.25), (-4.0, -4.0)] {
            let (a, b, c) = (inv[(0, 0)], inv[(0, 1)], inv[(1, 1)]);
            let fast = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy;
            let reference = mahalanobis2(&Vector2::new(dx, dy), &s).unwrap();
            assert!((fast - reference).abs() < 1e-12, "{fast} vs {reference}");
        }
    }

    /// Reference O(T*M) gating identical to the pre-optimisation code path.
    fn brute_force_pairs(t: &MultiTracker, meas: &[(f64, f64)]) -> Vec<(f64, usize, usize)> {
        use crate::kalman::mahalanobis2;
        use nalgebra::Vector2;
        let mut out = vec![];
        for (ti, trk) in t.tracks.iter().enumerate() {
            let (z, s) = trk.kf.innovation();
            for (mi, &(mx, my)) in meas.iter().enumerate() {
                if let Some(d2) = mahalanobis2(&(Vector2::new(mx, my) - z), &s) {
                    if d2 <= t.cfg.gate_d2 {
                        out.push((d2 + s.determinant().ln(), ti, mi));
                    }
                }
            }
        }
        out.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        out
    }

    #[test]
    fn windowed_gating_matches_brute_force() {
        let scen = crate::scenario::generate(3, 60, 30, 30, 0.1, 2.0, 300.0);
        let mut t = tracker();
        for m in &scen {
            // Predict a clone exactly as step() will, then compare gated pairs.
            let mut pre = t.clone();
            for trk in &mut pre.tracks {
                trk.kf.predict();
            }
            let want = brute_force_pairs(&pre, m);
            t.step(m).unwrap();
            let got = &t.pairs;
            assert_eq!(got.len(), want.len());
            for (g, w) in got.iter().zip(&want) {
                assert_eq!((g.1, g.2), (w.1, w.2));
                assert!((g.0 - w.0).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn empty_input_is_noop() {
        let mut t = tracker();
        assert!(t.step(&[]).unwrap().is_empty());
    }
}
