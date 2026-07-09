# rust_gbm

Zero-config gradient boosting for regression and classification.

Pure Rust, no dependencies, histogram binned. Compiles to WebAssembly.

```toml
[dependencies]
rust_gbm = "0.1"
```

## Use

`x` is a flat row-major slice of `n_rows * n_features` values.

```rust
use rust_gbm::{GbmRegressor, GbmClassifier};

// Regression
let reg = GbmRegressor::fit(&x, &y, n_rows, n_features);
let yhat: Vec<f32> = reg.predict(&x, n_rows);

// Classification. `labels` holds class indices as f32.
// Binary log-loss or multiclass softmax is selected automatically.
let clf = GbmClassifier::fit(&x, &labels, n_rows, n_features);
let class: usize = clf.predict_row(&row);
let probs: Vec<f32> = clf.predict_proba_row(&row);
```

The defaults are meant to work untuned. Override them with `fit_with`:

```rust
use rust_gbm::GbmParams;

let p = GbmParams { n_estimators: 400, learning_rate: 0.03, ..Default::default() };
let reg = GbmRegressor::fit_with(&x, &y, n_rows, n_features, &p);
```

| Param | Default |
| --- | --- |
| `n_estimators` | 200 |
| `learning_rate` | 0.05 |
| `max_depth` | 3 |
| `min_samples_leaf` | 1 |
| `subsample` | 0.8 |
| `max_bins` | 128 |
| `lambda` | 1.0 |

## WebAssembly

The `wasm` feature adds wasm-bindgen exports: `gbm_regress`, `gbm_classify`,
`gbm_classify_proba`. It is off by default so the core stays dependency free.

```bash
wasm-pack build --features wasm
```

## Performance

Native x86, release build, single threaded. `cargo run --release --example bench` reproduces this.

| rows | features | fit | predict/row |
| --- | --- | --- | --- |
| 100 | 8 | 3.3 ms | ~1 us |
| 1,000 | 16 | 20 ms | ~1 us |
| 10,000 | 16 | 187 ms | ~1 us |
| 50,000 | 32 | 2.6 s | ~1 us |

## Limits

- Numeric features only. No NaN handling. Encode categoricals upstream.
- Single threaded.
- No model serialization.

## License

Apache-2.0. See [LICENSE](LICENSE).
