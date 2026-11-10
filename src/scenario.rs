//! Deterministic, dependency-free scenario generator used by Criterion
//! benchmarks (the Python simulator remains the source of the regression
//! fixtures). Uses SplitMix64 + Box–Muller so results are identical on every
//! platform for a given seed.

pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(f64::MIN_POSITIVE);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// `n_targets` constant-velocity targets on a grid, `clutter` uniform false
/// alarms per step, Gaussian position noise `pos_std`. Returns one
/// measurement list per step (shuffled, no IDs).
pub fn generate(
    seed: u64,
    steps: usize,
    n_targets: usize,
    clutter: usize,
    dt: f64,
    pos_std: f64,
    area: f64,
) -> Vec<Vec<(f64, f64)>> {
    let mut rng = SplitMix64::new(seed);
    let mut targets: Vec<[f64; 4]> = (0..n_targets)
        .map(|_| {
            [
                (rng.uniform() * 2.0 - 1.0) * area * 0.8,
                (rng.uniform() * 2.0 - 1.0) * area * 0.8,
                rng.normal() * 5.0,
                rng.normal() * 5.0,
            ]
        })
        .collect();
    let mut out = Vec::with_capacity(steps);
    for _ in 0..steps {
        let mut m: Vec<(f64, f64)> = targets
            .iter()
            .map(|t| (t[0] + rng.normal() * pos_std, t[1] + rng.normal() * pos_std))
            .collect();
        for _ in 0..clutter {
            m.push((
                (rng.uniform() * 2.0 - 1.0) * area,
                (rng.uniform() * 2.0 - 1.0) * area,
            ));
        }
        for i in (1..m.len()).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            m.swap(i, j);
        }
        out.push(m);
        for t in &mut targets {
            t[0] += t[2] * dt;
            t[1] += t[3] * dt;
        }
    }
    out
}
