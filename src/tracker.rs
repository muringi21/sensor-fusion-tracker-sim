//! Multi-target tracker: predict → gate → greedy NN associate → update →
//! spawn → prune.

use crate::kalman::{mahalanobis2, KalmanError, KalmanFilterCV};
use nalgebra::{Vector2, Vector4};

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
        Self { dt: 0.1, q: 1.0, r_pos: 4.0, gate_d2: 9.21, max_missed: 5 }
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

        // Gating: candidate (d2, track, measurement) pairs.
        self.pairs.clear();
        for (ti, trk) in self.tracks.iter().enumerate() {
            let (z_pred, s) = trk.kf.innovation();
            for (mi, &(mx, my)) in measurements.iter().enumerate() {
                let y = Vector2::new(mx, my) - z_pred;
                if let Some(d2) = mahalanobis2(&y, &s) {
                    if d2 <= self.cfg.gate_d2 {
                        // Rank by negative log-likelihood (d2 + ln|S|), not raw
                        // d2: raw d2 favours young, high-covariance tracks, which
                        // then steal detections from established tracks.
                        self.pairs.push((d2 + s.determinant().ln(), ti, mi));
                    }
                }
            }
        }

        // Greedy nearest-neighbour: lowest d2 first. Ties broken by indices so
        // the result is fully deterministic.
        self.pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
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
                self.tracks.push(Track { id: self.next_id, kf, missed: 0, hits: 1 });
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
            let m = if k % 2 == 0 { [(x, 0.0), (x, 100.0)] } else { [(x, 100.0), (x, 0.0)] };
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
    fn empty_input_is_noop() {
        let mut t = tracker();
        assert!(t.step(&[]).unwrap().is_empty());
    }
}
