//! Criterion benchmarks for the core tracking loop.
//!
//! Each iteration runs a full fresh scenario (`STEPS` calls to
//! `MultiTracker::step`) so track spawning, association and pruning are all
//! included. Scenarios are generated once, outside the timed region, from a
//! fixed seed.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tracker_core::{scenario, MultiTracker, TrackerConfig};

const STEPS: usize = 100;

fn run(meas: &[Vec<(f64, f64)>], cfg: TrackerConfig) -> usize {
    let mut t = MultiTracker::new(cfg);
    let mut n = 0;
    for m in meas {
        n += t.step(black_box(m)).unwrap().len();
    }
    n
}

fn bench_tracking_loop(c: &mut Criterion) {
    let cfg = TrackerConfig::default();
    let mut g = c.benchmark_group("tracking_loop");
    // (targets, clutter per step)
    for &(n_targets, clutter) in &[(2usize, 0usize), (20, 20), (100, 100), (200, 200)] {
        let meas = scenario::generate(42, STEPS, n_targets, clutter, cfg.dt, 2.0, 2000.0);
        g.throughput(Throughput::Elements(STEPS as u64));
        g.bench_with_input(
            BenchmarkId::from_parameter(format!("{n_targets}t_{clutter}c")),
            &meas,
            |b, meas| b.iter(|| run(meas, cfg)),
        );
    }
    g.finish();
}

criterion_group!(benches, bench_tracking_loop);
criterion_main!(benches);
