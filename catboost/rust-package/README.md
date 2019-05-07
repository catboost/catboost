CatBoost Rust Package
======================

### Basic usage example

1. Checkout CatBoost repository.

2. Add a dependency to your Cargo.toml:
```
[dependencies]
catboost = { path = somepath/catboost/catboost/rust-package", version = "0.1"}
```
Where `somepath/catboost` is path to the repository root.

3. Now you can apply pretrained model in your code:
```rust
// Bring catboost module into the scope
use catboost;

fn main() {
    // Load the trained model
    let model = catboost::Model::load("tmp/model.bin").unwrap();

    println!("Number of cat features {}", model.get_cat_features_count());
    println!("Number of float features {}", model.get_float_features_count());

    // Apply the model
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
