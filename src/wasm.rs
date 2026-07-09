//! Browser bindings for the Sheets extension (enable with `--features wasm`). Same fit-on-support /
//! predict-query flow as the Postgres extension, over `wasm-bindgen`. Flat arrays are row-major
//! `[n_rows * n_feat]`; JS passes `Float32Array` and gets one back. Data never leaves the browser.
//!
//! ```js
//! import init, { gbm_regress, gbm_classify } from "./rust_gbm.js";
//! await init();
//! const preds = gbm_regress(supportX, supportY, queryX, nFeat); // Float32Array[n_query]
//! const cls   = gbm_classify(supportX, supportLabels, queryX, nFeat); // Uint32Array[n_query]
//! ```
use wasm_bindgen::prelude::*;

use crate::{GbmClassifier, GbmRegressor};

/// Regression: fit on the labeled (support) rows, predict the (query) rows. Returns `[n_query]`.
#[wasm_bindgen]
pub fn gbm_regress(
    support_x: &[f32],
    support_y: &[f32],
    query_x: &[f32],
    n_feat: usize,
) -> Vec<f32> {
    let ns = support_y.len();
    let m = GbmRegressor::fit(support_x, support_y, ns, n_feat);
    let nq = if n_feat > 0 {
        query_x.len() / n_feat
    } else {
        0
    };
    m.predict(query_x, nq)
}

/// Classification (auto binary/multiclass from labels 0..K-1): predicted class id per query row.
#[wasm_bindgen]
pub fn gbm_classify(
    support_x: &[f32],
    support_labels: &[f32],
    query_x: &[f32],
    n_feat: usize,
) -> Vec<u32> {
    let ns = support_labels.len();
    let m = GbmClassifier::fit(support_x, support_labels, ns, n_feat);
    let nq = if n_feat > 0 {
        query_x.len() / n_feat
    } else {
        0
    };
    (0..nq)
        .map(|i| m.predict_row(&query_x[i * n_feat..i * n_feat + n_feat]) as u32)
        .collect()
}

/// Classification probabilities, flattened `[n_query * n_classes]` (row-major).
#[wasm_bindgen]
pub fn gbm_classify_proba(
    support_x: &[f32],
    support_labels: &[f32],
    query_x: &[f32],
    n_feat: usize,
) -> Vec<f32> {
    let ns = support_labels.len();
    let m = GbmClassifier::fit(support_x, support_labels, ns, n_feat);
    let nq = if n_feat > 0 {
        query_x.len() / n_feat
    } else {
        0
    };
    let mut out = Vec::with_capacity(nq * m.n_classes());
    for i in 0..nq {
        out.extend(m.predict_proba_row(&query_x[i * n_feat..i * n_feat + n_feat]));
    }
    out
}
