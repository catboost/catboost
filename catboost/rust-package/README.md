CatBoost Rust Package
======================

### Basic usage example

```rust
use catboost;

fn main() {
    let model = catboost::Model::load("tmp/model.bin").unwrap();

    println!("Number of cat features {:?}", model.get_cat_features_count());
    println!("Number of float features {:?}", model.get_float_features_count());

    let prediction = model
        .calc_model_prediction(
            vec![
                vec![-10.0, 5.0, 753.0],
                vec![30.0, 1.0, 760.0],
                vec![40.0, 0.1, 705.0],
            ],
            vec![
                vec![String::from("north")],
                vec![String::from("south")],
                vec![String::from("south")],
            ],
        )
        .unwrap();
    println!("Prediction {:?}", prediction);
}
```

### Documentation
Run `cargo doc --open` in `catboost/rust-package` directory.

### Tests

Run `cargo test` in `catboost/rust-package` directory.

To run tests with sanitizer, uncomment line `"--sanitize=address",` in `catboost/rust-package/catboost-sys/build.rs` and run `RUSTFLAGS="-Z sanitizer=address" cargo +nightly test`.