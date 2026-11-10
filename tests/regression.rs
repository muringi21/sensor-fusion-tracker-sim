//! Fixed-seed simulation regression tests.
//!
//! Fixtures in `tests/fixtures/*.json` are produced by the seeded Python
//! simulator (`scripts/gen_fixtures.py`). Two layers of checks:
//!
//! 1. **Accuracy envelopes** against ground truth (see "Tolerances" in the
//!    README). These encode what "correct tracking" means for each scenario and
//!    are loose enough to survive legitimate numerical changes.
//! 2. **Golden snapshot** of the tracker's exact output (`tests/golden/`). Any
//!    behavioural change — including from a performance optimisation — fails
//!    here unless outputs agree to `GOLDEN_ABS_TOL`. Regenerate deliberately
//!    with `UPDATE_GOLDEN=1 cargo test --test regression`.

use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};
use tracker_core::{MultiTracker, TrackerConfig};

/// Floating-point tolerance for golden comparisons. The pipeline is
/// deterministic, so on one machine outputs are bit-identical; 1e-9 m leaves
/// room for FMA / libm differences across CPUs and compilers.
const GOLDEN_ABS_TOL: f64 = 1e-9;
/// Steps ignored before scoring accuracy (filter starts with zero velocity).
const WARMUP: usize = 10;
/// A track "covers" a target at step k if its position is within this many metres.
const MATCH_RADIUS: f64 = 10.0;

#[derive(Deserialize)]
struct Fixture {
    name: String,
    dt: f64,
    truth: Vec<Vec<[f64; 4]>>,
    measurements: Vec<Vec<[f64; 2]>>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Golden {
    tracks_per_step: Vec<usize>,
    next_ids_seen: u64,
    final_tracks: Vec<(u64, [f64; 4])>,
    /// Sum over steps/tracks of x+y+vx+vy: cheap fingerprint of the full trajectory.
    checksum: f64,
}

#[derive(Debug)]
struct Metrics {
    pos_rmse: Vec<f64>,
    vel_rmse: Vec<f64>,
    coverage: Vec<f64>,
    id_switches: Vec<usize>,
    max_tracks: usize,
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn load(name: &str) -> Fixture {
    let p = root().join("tests/fixtures").join(format!("{name}.json"));
    serde_json::from_str(&fs::read_to_string(&p).unwrap()).unwrap()
}

fn run(fx: &Fixture) -> (Metrics, Golden) {
    let cfg = TrackerConfig {
        dt: fx.dt,
        max_missed: 6,
        ..TrackerConfig::default()
    };
    let mut trk = MultiTracker::new(cfg);
    let nt = fx.truth.len();
    let (mut se_p, mut se_v, mut cnt, mut covered) = (
        vec![0.0; nt],
        vec![0.0; nt],
        vec![0usize; nt],
        vec![0usize; nt],
    );
    let mut last_id: Vec<Option<u64>> = vec![None; nt];
    let mut switches = vec![0usize; nt];
    let mut g = Golden {
        tracks_per_step: vec![],
        next_ids_seen: 0,
        final_tracks: vec![],
        checksum: 0.0,
    };
    let mut max_tracks = 0;

    for (k, m) in fx.measurements.iter().enumerate() {
        let meas: Vec<(f64, f64)> = m.iter().map(|p| (p[0], p[1])).collect();
        let tracks = trk.step(&meas).unwrap();
        g.tracks_per_step.push(tracks.len());
        max_tracks = max_tracks.max(tracks.len());
        for t in tracks {
            g.next_ids_seen = g.next_ids_seen.max(t.id);
            g.checksum += t.kf.state().iter().sum::<f64>();
        }
        if k < WARMUP {
            continue;
        }
        for (ti, gt) in fx.truth.iter().enumerate() {
            let s = gt[k];
            let best = tracks
                .iter()
                .filter(|t| t.hits >= 3)
                .map(|t| {
                    (
                        t,
                        ((t.kf.x[0] - s[0]).powi(2) + (t.kf.x[1] - s[1]).powi(2)).sqrt(),
                    )
                })
                .min_by(|a, b| a.1.total_cmp(&b.1));
            if let Some((t, d)) = best.filter(|b| b.1 <= MATCH_RADIUS) {
                covered[ti] += 1;
                se_p[ti] += d * d;
                se_v[ti] += (t.kf.x[2] - s[2]).powi(2) + (t.kf.x[3] - s[3]).powi(2);
                cnt[ti] += 1;
                if last_id[ti].is_some_and(|id| id != t.id) {
                    switches[ti] += 1;
                }
                last_id[ti] = Some(t.id);
            }
        }
    }
    g.final_tracks = trk.tracks().iter().map(|t| (t.id, t.kf.state())).collect();
    let scored = (fx.measurements.len() - WARMUP) as f64;
    let m = Metrics {
        pos_rmse: (0..nt)
            .map(|i| (se_p[i] / cnt[i].max(1) as f64).sqrt())
            .collect(),
        vel_rmse: (0..nt)
            .map(|i| (se_v[i] / cnt[i].max(1) as f64).sqrt())
            .collect(),
        coverage: covered.iter().map(|&c| c as f64 / scored).collect(),
        id_switches: switches,
        max_tracks,
    };
    (m, g)
}

fn check_golden(name: &str, g: &Golden) {
    let p = root().join("tests/golden").join(format!("{name}.json"));
    if std::env::var_os("UPDATE_GOLDEN").is_some() || !p.exists() {
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(&p, serde_json::to_string_pretty(g).unwrap() + "\n").unwrap();
        eprintln!("updated golden {}", p.display());
        return;
    }
    let want: Golden = serde_json::from_str(&fs::read_to_string(&p).unwrap()).unwrap();
    assert_eq!(
        g.tracks_per_step, want.tracks_per_step,
        "{name}: track count trajectory changed"
    );
    assert_eq!(
        g.next_ids_seen, want.next_ids_seen,
        "{name}: track ids changed"
    );
    assert_eq!(g.final_tracks.len(), want.final_tracks.len());
    for ((id, s), (wid, ws)) in g.final_tracks.iter().zip(&want.final_tracks) {
        assert_eq!(id, wid);
        for (a, b) in s.iter().zip(ws) {
            assert!(
                (a - b).abs() <= GOLDEN_ABS_TOL,
                "{name}: track {id} state {a} vs {b}"
            );
        }
    }
    let tol = GOLDEN_ABS_TOL * g.tracks_per_step.iter().sum::<usize>() as f64 * 4.0;
    assert!(
        (g.checksum - want.checksum).abs() <= tol.max(1e-6),
        "{name}: checksum {} vs {}",
        g.checksum,
        want.checksum
    );
}

struct Envelope {
    max_pos_rmse: f64,
    max_vel_rmse: f64,
    min_coverage: f64,
    max_id_switches: usize,
    max_tracks: usize,
}

fn scenario(name: &str, env: Envelope) {
    let fx = load(name);
    assert_eq!(fx.name, name);
    let (m, g) = run(&fx);
    eprintln!("{name}: {m:?}");
    for i in 0..fx.truth.len() {
        assert!(
            m.pos_rmse[i] <= env.max_pos_rmse,
            "{name} target {i}: pos RMSE {:.3}",
            m.pos_rmse[i]
        );
        assert!(
            m.vel_rmse[i] <= env.max_vel_rmse,
            "{name} target {i}: vel RMSE {:.3}",
            m.vel_rmse[i]
        );
        assert!(
            m.coverage[i] >= env.min_coverage,
            "{name} target {i}: coverage {:.3}",
            m.coverage[i]
        );
        assert!(
            m.id_switches[i] <= env.max_id_switches,
            "{name} target {i}: {} id switches",
            m.id_switches[i]
        );
    }
    assert!(
        m.max_tracks <= env.max_tracks,
        "{name}: {} simultaneous tracks",
        m.max_tracks
    );
    check_golden(name, &g);
}

// Envelopes are documented in README.md → "Tolerances".

#[test]
fn two_targets_clean() {
    scenario(
        "two_targets_clean",
        Envelope {
            max_pos_rmse: 1.2,
            max_vel_rmse: 1.5,
            min_coverage: 1.0,
            max_id_switches: 0,
            max_tracks: 3,
        },
    );
}

#[test]
fn two_targets_clutter() {
    scenario(
        "two_targets_clutter",
        Envelope {
            max_pos_rmse: 1.2,
            max_vel_rmse: 1.0,
            min_coverage: 0.99,
            max_id_switches: 0,
            max_tracks: 40,
        },
    );
}

#[test]
fn three_targets_crossing() {
    scenario(
        "three_targets_crossing",
        Envelope {
            max_pos_rmse: 1.5,
            max_vel_rmse: 2.5,
            min_coverage: 0.98,
            max_id_switches: 2,
            max_tracks: 12,
        },
    );
}

#[test]
fn simulation_is_deterministic() {
    let fx = load("two_targets_clutter");
    let (_, a) = run(&fx);
    let (_, b) = run(&fx);
    assert_eq!(a, b);
}
