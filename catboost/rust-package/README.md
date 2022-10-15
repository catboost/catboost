CatBoost Rust Package
======================

### Differences with official rust package
* The official `catboost-sys` one attempts to rebuild the shared library, whereas this one downloads it from the github release page.
* The `build.rs` script is rewritten to also work for M1 macs (same strategy, downloading the shared library).
* Also marked the Model as `Send` so that it can be used across threads, due to the documentation stating it's thread safe. Note that this is not yet extensively tested though.
* As of the present the catboost version is hardcoded, it is currently 1.0.6.

### TODO
* It seems slightly excessive to have to fork the whole repo just to maintain the code for the rust bindings
* It also seems excessive to have to clone the whole repo, as all we need is `wrapper.h` and the files in the `model_interface` folder.
* Perhaps a better idea is to split out the repo, similar to `onnxruntime-rs`

### Basic usage example

1. Add a dependency to your Cargo.toml:
```
[dependencies]
catboost-rs = "0.1.3"
```
2. To use catboost, it assumes the shared libraries are available. You will need to download the shared library from the official [releases page](https://github.com/catboost/catboost/releases). If you are using linux, download `libcatboostmodel.so`. If you are using Mac, download `libcatboostmodel.dylib`. As of the present, only version 1.0.6 is supported.
3. Move these libraries to `/usr/lib` 
4. Now you can apply pretrained model in your code:
```rust
// Bring catboost module into the scope
use catboost_rs as catboost;

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
