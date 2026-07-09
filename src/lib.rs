//! Zero-config gradient boosting for regression and classification.
//!
//! Histogram-binned, second-order (gradient/hessian) boosting with CART trees.
//! Pure Rust, no dependencies, so it builds for native targets and for WebAssembly.
//!
//! - Regression: [`GbmRegressor`] (squared error).
//! - Classification: [`GbmClassifier`] (binary log-loss, or multiclass softmax, auto-detected).
//!
//! Binning makes a fit O(rows * bins) instead of sorting at every split. There are no threads,
//! filesystem, clock or rng-crate dependencies, so it is WASM-safe and deterministic (a small
//! SplitMix64 drives subsampling).
//!
//! ```
//! use rust_gbm::{GbmRegressor, GbmClassifier};
//! let x = [0.0f32, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0];
//! let m = GbmRegressor::fit(&x, &[1.0, 1.0, 2.0, 0.0], 4, 2);
//! let _ = m.predict_row(&[1.0, 1.0]);
//! let c = GbmClassifier::fit(&x, &[0.0, 0.0, 1.0, 0.0], 4, 2);
//! let _ = c.predict_row(&[1.0, 1.0]);
//! ```

use std::cmp::Ordering;

#[cfg(feature = "wasm")]
mod wasm;

/// Hyperparameters. `Default` is tuned to "just works" - you should never need to touch these.
#[derive(Clone, Debug)]
pub struct GbmParams {
    pub n_estimators: usize,
    pub learning_rate: f32,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub subsample: f32,
    pub max_bins: usize, // histogram resolution (<=256)
    pub lambda: f32,     // L2 leaf regularization
    pub seed: u64,
}

impl Default for GbmParams {
    fn default() -> Self {
        GbmParams {
            n_estimators: 200,
            learning_rate: 0.05,
            max_depth: 3,
            min_samples_leaf: 1,
            subsample: 0.8,
            max_bins: 128,
            lambda: 1.0,
            seed: 0x9E37_79B9_7F4A_7C15,
        }
    }
}

// ------------------------------ feature binning ------------------------------

/// Monotonic per-feature binning (quantile edges). Fit once on the support rows; reused to bin queries.
struct BinMapper {
    edges: Vec<Vec<f32>>, // edges[f] ascending; a value maps to `partition_point(e < v)` in 0..=edges.len()
    n_features: usize,
}

impl BinMapper {
    fn fit(x: &[f32], n: usize, d: usize, max_bins: usize) -> Self {
        let mut edges = Vec::with_capacity(d);
        for f in 0..d {
            let mut vals: Vec<f32> = (0..n).map(|i| x[i * d + f]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let mut distinct: Vec<f32> = Vec::new();
            for v in vals {
                if distinct.last().map_or(true, |&l| l != v) {
                    distinct.push(v);
                }
            }
            let e = if distinct.len() <= max_bins {
                (0..distinct.len().saturating_sub(1))
                    .map(|k| 0.5 * (distinct[k] + distinct[k + 1]))
                    .collect()
            } else {
                // max_bins-1 quantile edges over the distinct values
                (1..max_bins)
                    .map(|k| {
                        let idx = k * distinct.len() / max_bins;
                        0.5 * (distinct[idx - 1] + distinct[idx.min(distinct.len() - 1)])
                    })
                    .collect()
            };
            edges.push(e);
        }
        BinMapper {
            edges,
            n_features: d,
        }
    }

    #[inline]
    fn bin_val(&self, f: usize, v: f32) -> u8 {
        self.edges[f].partition_point(|&e| e < v) as u8
    }

    fn nbins(&self, f: usize) -> usize {
        self.edges[f].len() + 1
    }

    fn transform(&self, x: &[f32], n: usize) -> Vec<u8> {
        let d = self.n_features;
        let mut b = vec![0u8; n * d];
        for i in 0..n {
            for f in 0..d {
                b[i * d + f] = self.bin_val(f, x[i * d + f]);
            }
        }
        b
    }
}

// ------------------------------ histogram tree ------------------------------

#[derive(Clone)]
struct Node {
    feature: u32, // u32::MAX => leaf
    bin_thr: u8,  // go left if bin <= bin_thr
    left: u32,
    right: u32,
    value: f32,
}

struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    #[inline]
    fn predict_binned(&self, row: &[u8]) -> f32 {
        let mut i = 0usize;
        loop {
            let nd = &self.nodes[i];
            if nd.feature == u32::MAX {
                return nd.value;
            }
            i = if row[nd.feature as usize] <= nd.bin_thr {
                nd.left as usize
            } else {
                nd.right as usize
            };
        }
    }
}

struct Grow<'a> {
    binned: &'a [u8],
    grad: &'a [f32],
    hess: &'a [f32],
    mapper: &'a BinMapper,
    d: usize,
    p: &'a GbmParams,
}

impl Grow<'_> {
    fn leaf_value(&self, idx: &[usize]) -> f32 {
        let (mut g, mut h) = (0.0f32, 0.0f32);
        for &i in idx {
            g += self.grad[i];
            h += self.hess[i];
        }
        -g / (h + self.p.lambda)
    }

    fn build(&self, idx: &[usize]) -> Tree {
        let mut nodes = Vec::new();
        self.build_node(idx, 0, &mut nodes);
        Tree { nodes }
    }

    fn build_node(&self, idx: &[usize], depth: usize, nodes: &mut Vec<Node>) -> u32 {
        let me = nodes.len() as u32;
        nodes.push(Node {
            feature: u32::MAX,
            bin_thr: 0,
            left: 0,
            right: 0,
            value: self.leaf_value(idx),
        });
        let min_leaf = self.p.min_samples_leaf.max(1);
        if depth >= self.p.max_depth || idx.len() < 2 * min_leaf {
            return me;
        }
        if let Some((feat, thr)) = self.best_split(idx) {
            let (mut left, mut right) = (Vec::new(), Vec::new());
            for &i in idx {
                if self.binned[i * self.d + feat as usize] <= thr {
                    left.push(i);
                } else {
                    right.push(i);
                }
            }
            if left.len() < min_leaf || right.len() < min_leaf {
                return me;
            }
            let l = self.build_node(&left, depth + 1, nodes);
            let r = self.build_node(&right, depth + 1, nodes);
            let nd = &mut nodes[me as usize];
            nd.feature = feat;
            nd.bin_thr = thr;
            nd.left = l;
            nd.right = r;
        }
        me
    }

    /// Best (feature, bin threshold) by second-order gain: sum(g)^2 / (sum(h) + lambda) over the children.
    fn best_split(&self, idx: &[usize]) -> Option<(u32, u8)> {
        let (mut gt, mut ht) = (0.0f32, 0.0f32);
        for &i in idx {
            gt += self.grad[i];
            ht += self.hess[i];
        }
        let base = gt * gt / (ht + self.p.lambda);
        let min_leaf = self.p.min_samples_leaf.max(1);
        let mut best: Option<(f32, u32, u8)> = None;

        for f in 0..self.d {
            let nb = self.mapper.nbins(f);
            let mut hg = vec![0.0f32; nb];
            let mut hh = vec![0.0f32; nb];
            let mut hc = vec![0u32; nb];
            for &i in idx {
                let b = self.binned[i * self.d + f] as usize;
                hg[b] += self.grad[i];
                hh[b] += self.hess[i];
                hc[b] += 1;
            }
            let (mut gl, mut hl, mut cl) = (0.0f32, 0.0f32, 0u32);
            for b in 0..nb - 1 {
                gl += hg[b];
                hl += hh[b];
                cl += hc[b];
                if (cl as usize) < min_leaf || (idx.len() - cl as usize) < min_leaf {
                    continue;
                }
                let gr = gt - gl;
                let hr = ht - hl;
                let gain = gl * gl / (hl + self.p.lambda) + gr * gr / (hr + self.p.lambda) - base;
                if best.map_or(true, |(bg, _, _)| gain > bg) {
                    best = Some((gain, f as u32, b as u8));
                }
            }
        }
        let (g, f, t) = best?;
        if g > 1e-6 {
            Some((f, t))
        } else {
            None
        }
    }
}

// ------------------------------ objectives ------------------------------

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

enum Obj {
    L2,
    Binary,
    Multiclass(usize),
}

impl Obj {
    fn n_out(&self) -> usize {
        match self {
            Obj::Multiclass(k) => *k,
            _ => 1,
        }
    }

    fn init_base(&self, y: &[f32]) -> Vec<f32> {
        match self {
            Obj::L2 => vec![y.iter().sum::<f32>() / y.len().max(1) as f32],
            Obj::Binary => {
                let mu = (y.iter().sum::<f32>() / y.len().max(1) as f32).clamp(1e-4, 1.0 - 1e-4);
                vec![(mu / (1.0 - mu)).ln()]
            }
            Obj::Multiclass(k) => {
                let mut cnt = vec![1.0f32; *k]; // Laplace
                for &v in y {
                    cnt[v as usize] += 1.0;
                }
                let tot: f32 = cnt.iter().sum();
                cnt.iter().map(|c| (c / tot).ln()).collect()
            }
        }
    }

    /// Fill grad/hess `[n * n_out]` from raw scores `[n * n_out]` and labels/targets `y[n]`.
    fn grad_hess(&self, raw: &[f32], y: &[f32], grad: &mut [f32], hess: &mut [f32]) {
        let n = y.len();
        match self {
            Obj::L2 => {
                for i in 0..n {
                    grad[i] = raw[i] - y[i];
                    hess[i] = 1.0;
                }
            }
            Obj::Binary => {
                for i in 0..n {
                    let p = sigmoid(raw[i]);
                    grad[i] = p - y[i];
                    hess[i] = (p * (1.0 - p)).max(1e-6);
                }
            }
            Obj::Multiclass(k) => {
                for i in 0..n {
                    let r = &raw[i * k..i * k + k];
                    let m = r.iter().cloned().fold(f32::MIN, f32::max);
                    let mut den = 0.0f32;
                    let mut e = vec![0.0f32; *k];
                    for c in 0..*k {
                        e[c] = (r[c] - m).exp();
                        den += e[c];
                    }
                    let yi = y[i] as usize;
                    for c in 0..*k {
                        let p = e[c] / den;
                        grad[i * k + c] = p - if c == yi { 1.0 } else { 0.0 };
                        hess[i * k + c] = (p * (1.0 - p)).max(1e-6);
                    }
                }
            }
        }
    }
}

// ------------------------------ booster core ------------------------------

struct Booster {
    mapper: BinMapper,
    base: Vec<f32>,        // [n_out]
    trees: Vec<Vec<Tree>>, // [n_out][round]
    lr: f32,
    n_out: usize,
}

impl Booster {
    fn train(x: &[f32], y: &[f32], n: usize, d: usize, obj: Obj, p: &GbmParams) -> Self {
        assert_eq!(x.len(), n * d);
        assert_eq!(y.len(), n);
        let n_out = obj.n_out();
        let mapper = BinMapper::fit(x, n, d, p.max_bins.clamp(2, 256));
        let binned = mapper.transform(x, n);
        let base = obj.init_base(y);
        let mut raw = vec![0.0f32; n * n_out];
        for i in 0..n {
            for o in 0..n_out {
                raw[i * n_out + o] = base[o];
            }
        }
        let mut trees: Vec<Vec<Tree>> = (0..n_out)
            .map(|_| Vec::with_capacity(p.n_estimators))
            .collect();
        let mut grad = vec![0.0f32; n * n_out];
        let mut hess = vec![0.0f32; n * n_out];
        let mut rng = SplitMix64::new(p.seed);
        let sub_n = (((n as f32) * p.subsample).ceil() as usize).clamp(1, n.max(1));

        for _ in 0..p.n_estimators {
            obj.grad_hess(&raw, y, &mut grad, &mut hess);
            for o in 0..n_out {
                let go: Vec<f32> = (0..n).map(|i| grad[i * n_out + o]).collect();
                let ho: Vec<f32> = (0..n).map(|i| hess[i * n_out + o]).collect();
                let idx = sample_indices(n, sub_n, &mut rng);
                let grow = Grow {
                    binned: &binned,
                    grad: &go,
                    hess: &ho,
                    mapper: &mapper,
                    d,
                    p,
                };
                let tree = grow.build(&idx);
                for i in 0..n {
                    raw[i * n_out + o] +=
                        p.learning_rate * tree.predict_binned(&binned[i * d..i * d + d]);
                }
                trees[o].push(tree);
            }
        }
        Booster {
            mapper,
            base,
            trees,
            lr: p.learning_rate,
            n_out,
        }
    }

    /// Raw scores for one feature row (`[n_out]`).
    fn raw_row(&self, row: &[f32]) -> Vec<f32> {
        let d = self.mapper.n_features;
        let mut binned = vec![0u8; d];
        for f in 0..d {
            binned[f] = self.mapper.bin_val(f, row[f]);
        }
        let mut out = self.base.clone();
        for (o, acc) in out.iter_mut().enumerate().take(self.n_out) {
            for t in &self.trees[o] {
                *acc += self.lr * t.predict_binned(&binned);
            }
        }
        out
    }
}

// ------------------------------ public API ------------------------------

/// Zero-config gradient-boosting **regressor** (squared error).
pub struct GbmRegressor {
    b: Booster,
}

impl GbmRegressor {
    pub fn fit(x: &[f32], y: &[f32], n_rows: usize, n_features: usize) -> Self {
        Self::fit_with(x, y, n_rows, n_features, &GbmParams::default())
    }
    pub fn fit_with(x: &[f32], y: &[f32], n: usize, d: usize, p: &GbmParams) -> Self {
        GbmRegressor {
            b: Booster::train(x, y, n, d, Obj::L2, p),
        }
    }
    pub fn predict_row(&self, row: &[f32]) -> f32 {
        self.b.raw_row(row)[0]
    }
    pub fn predict(&self, x: &[f32], n_rows: usize) -> Vec<f32> {
        let d = self.b.mapper.n_features;
        (0..n_rows)
            .map(|i| self.predict_row(&x[i * d..i * d + d]))
            .collect()
    }
}

/// Zero-config gradient-boosting **classifier**. Binary (log-loss) or multiclass (softmax), auto-detected
/// from the labels (0..K-1). Predicts class probabilities and the argmax class.
pub struct GbmClassifier {
    b: Booster,
    n_classes: usize,
    binary: bool,
}

impl GbmClassifier {
    pub fn fit(x: &[f32], y: &[f32], n_rows: usize, n_features: usize) -> Self {
        Self::fit_with(x, y, n_rows, n_features, &GbmParams::default())
    }
    pub fn fit_with(x: &[f32], y: &[f32], n: usize, d: usize, p: &GbmParams) -> Self {
        let k = (y.iter().cloned().fold(0.0f32, f32::max) as usize) + 1;
        let (obj, binary) = if k <= 2 {
            (Obj::Binary, true)
        } else {
            (Obj::Multiclass(k), false)
        };
        GbmClassifier {
            b: Booster::train(x, y, n, d, obj, p),
            n_classes: k.max(2),
            binary,
        }
    }

    /// Class probabilities `[n_classes]` for one row.
    pub fn predict_proba_row(&self, row: &[f32]) -> Vec<f32> {
        let raw = self.b.raw_row(row);
        if self.binary {
            let p = sigmoid(raw[0]);
            vec![1.0 - p, p]
        } else {
            let m = raw.iter().cloned().fold(f32::MIN, f32::max);
            let e: Vec<f32> = raw.iter().map(|&r| (r - m).exp()).collect();
            let den: f32 = e.iter().sum();
            e.iter().map(|v| v / den).collect()
        }
    }

    /// Argmax class id for one row.
    pub fn predict_row(&self, row: &[f32]) -> usize {
        let p = self.predict_proba_row(row);
        let mut best = 0;
        for c in 1..p.len() {
            if p[c] > p[best] {
                best = c;
            }
        }
        best
    }

    pub fn n_classes(&self) -> usize {
        self.n_classes
    }
}

// ------------------------------ helpers ------------------------------

/// R (coefficient of determination) - for the products' "accuracy check".
pub fn r2_score(y_true: &[f32], y_pred: &[f32]) -> f32 {
    let n = y_true.len();
    if n == 0 {
        return 0.0;
    }
    let mean = y_true.iter().sum::<f32>() / n as f32;
    let (mut sr, mut st) = (0.0f32, 0.0f32);
    for i in 0..n {
        sr += (y_true[i] - y_pred[i]).powi(2);
        st += (y_true[i] - mean).powi(2);
    }
    if st == 0.0 {
        0.0
    } else {
        1.0 - sr / st
    }
}

/// Fraction of exact-match predictions.
pub fn accuracy(y_true: &[f32], y_pred_class: &[usize]) -> f32 {
    if y_true.is_empty() {
        return 0.0;
    }
    let c = (0..y_true.len())
        .filter(|&i| y_true[i] as usize == y_pred_class[i])
        .count();
    c as f32 / y_true.len() as f32
}

struct SplitMix64 {
    s: u64,
}
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64 { s: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.s = self.s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_bounded(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

fn sample_indices(n: usize, k: usize, rng: &mut SplitMix64) -> Vec<usize> {
    let mut a: Vec<usize> = (0..n).collect();
    let k = k.min(n);
    for i in 0..k {
        let j = i + rng.next_bounded(n - i);
        a.swap(i, j);
    }
    a.truncate(k);
    a
}

// ------------------------------ tests ------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn rng_unit(s: &mut u64) -> f32 {
        *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f32 / (1u64 << 53) as f32
    }

    fn feats(n: usize, d: usize, seed: u64) -> (Vec<f32>, u64) {
        let mut s = seed;
        let mut x = vec![0.0f32; n * d];
        for v in x.iter_mut() {
            *v = rng_unit(&mut s) * 2.0 - 1.0;
        }
        (x, s)
    }

    #[test]
    fn regression_nonlinear() {
        let (d, ntr, nte) = (5, 400, 200);
        let (xtr, _) = feats(ntr, d, 1);
        let (xte, _) = feats(nte, d, 2);
        let tgt = |x: &[f32]| (x[0] * 3.0).sin() + x[1] * x[1] + 0.5 * x[0] * x[2];
        let ytr: Vec<f32> = (0..ntr).map(|i| tgt(&xtr[i * d..i * d + d])).collect();
        let yte: Vec<f32> = (0..nte).map(|i| tgt(&xte[i * d..i * d + d])).collect();
        let m = GbmRegressor::fit(&xtr, &ytr, ntr, d);
        let r2 = r2_score(&yte, &m.predict(&xte, nte));
        println!("regression held-out R2 = {r2:.3}");
        assert!(r2 > 0.85, "R2 too low {r2}");
    }

    #[test]
    fn binary_classification() {
        let (d, ntr, nte) = (6, 500, 300);
        let (xtr, _) = feats(ntr, d, 3);
        let (xte, _) = feats(nte, d, 4);
        // nonlinear boundary
        let lab = |x: &[f32]| ((x[0] * x[0] + x[1] - 0.3 * x[2]) > 0.4) as usize as f32;
        let ytr: Vec<f32> = (0..ntr).map(|i| lab(&xtr[i * d..i * d + d])).collect();
        let yte: Vec<f32> = (0..nte).map(|i| lab(&xte[i * d..i * d + d])).collect();
        let m = GbmClassifier::fit(&xtr, &ytr, ntr, d);
        let pred: Vec<usize> = (0..nte)
            .map(|i| m.predict_row(&xte[i * d..i * d + d]))
            .collect();
        let acc = accuracy(&yte, &pred);
        println!("binary held-out acc = {acc:.3}");
        assert_eq!(m.n_classes(), 2);
        assert!(acc > 0.9, "acc too low {acc}");
    }

    #[test]
    fn multiclass_classification() {
        let (d, ntr, nte) = (6, 600, 300);
        let (xtr, _) = feats(ntr, d, 5);
        let (xte, _) = feats(nte, d, 6);
        let lab = |x: &[f32]| -> f32 {
            let s = x[0] + x[1];
            if s < -0.4 {
                0.0
            } else if s < 0.4 {
                1.0
            } else {
                2.0
            }
        };
        let ytr: Vec<f32> = (0..ntr).map(|i| lab(&xtr[i * d..i * d + d])).collect();
        let yte: Vec<f32> = (0..nte).map(|i| lab(&xte[i * d..i * d + d])).collect();
        let m = GbmClassifier::fit(&xtr, &ytr, ntr, d);
        let pred: Vec<usize> = (0..nte)
            .map(|i| m.predict_row(&xte[i * d..i * d + d]))
            .collect();
        let acc = accuracy(&yte, &pred);
        println!(
            "multiclass held-out acc = {acc:.3} ({} classes)",
            m.n_classes()
        );
        assert_eq!(m.n_classes(), 3);
        assert!(acc > 0.9, "acc too low {acc}");
    }
}
