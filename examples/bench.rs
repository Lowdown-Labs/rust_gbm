// Query-time cost of fit + predict across realistic support-set sizes. Native bench only.
use rust_gbm::{r2_score, GbmRegressor};
use std::time::Instant;

fn splitmix(s: &mut u64) -> f32 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) >> 11) as f32 / (1u64 << 53) as f32
}

fn gen(n: usize, d: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut s = seed;
    let mut x = vec![0.0; n * d];
    let mut y = vec![0.0; n];
    for i in 0..n {
        for j in 0..d {
            x[i * d + j] = splitmix(&mut s) * 2.0 - 1.0;
        }
        let (a, b, c) = (x[i * d], x[i * d + 1], x[i * d + 2 % d]);
        y[i] = (a * 3.0).sin() + b * b + 0.5 * a * c + (splitmix(&mut s) - 0.5) * 0.1;
    }
    (x, y)
}

fn main() {
    println!(
        "{:>8} {:>6} | {:>10} | {:>12} | {:>8}",
        "rows", "feats", "fit (ms)", "predict/row(us)", "R2"
    );
    for &(n, d) in &[(100usize, 8usize), (1_000, 16), (10_000, 16), (50_000, 32)] {
        let (x, y) = gen(n, d, 42);
        let (xte, yte) = gen(2_000.min(n), d, 7);

        let t0 = Instant::now();
        let m = GbmRegressor::fit(&x, &y, n, d);
        let fit_ms = t0.elapsed().as_secs_f64() * 1e3;

        let t1 = Instant::now();
        let pred = m.predict(&xte, yte.len());
        let per_row_us = t1.elapsed().as_secs_f64() * 1e6 / yte.len() as f64;

        let r2 = r2_score(&yte, &pred);
        println!("{n:>8} {d:>6} | {fit_ms:>10.1} | {per_row_us:>12.2} | {r2:>8.3}");
    }
}
