CatBoost Rust Package
======================

### Prerequisites

The minimal supported Rust version is 1.64.0 .

1. CatBoost Rust package uses `catboost-sys` crate inside that is a wrapper around [`libcatboostmodel` library](https://catboost.ai/docs/en/concepts/c-plus-plus-api_dynamic-c-pluplus-wrapper) with an exposed C API.
In order to build it some environment setup is necessary. Modern versions of CatBoost use CMake build system, build environment setup for CMake is described [here](https://catboost.ai/docs/en/installation/build-environment-setup-for-cmake), CatBoost versions before 1.2 used Ya Make build system, build environment setup for YaMake is described [here](https://catboost.ai/docs/en/installation/build-environment-setup-for-ya-make).

2. This package uses [bindgen](https://rust-lang.github.io/rust-bindgen/introduction.html) to generate bindings that has [its own requirements](https://rust-lang.github.io/rust-bindgen/requirements.html).

3. If you run into issues on Windows like `stdbool.h not found` you should set up the environment for the appropriate version of Microsoft Visual Studio C++ toolset before running `cargo` so `bindgen` can find this header.

    It can be done with these commands (in PowerShell):

    ```
    Import-Module "<VSDiskDrive>:\Program Files\Microsoft Visual Studio\<VSVersion>\<Edition>\Common7\Tools\Microsoft.VisualStudio.DevShell.dll
    Enter-VsDevShell -VsInstallPath "<VSDiskDrive>:\Program Files\Microsoft Visual Studio\<VSVersion>\<Edition>" -DevCmdArguments "-vcvars_ver=<VCVars>"
    ```

    where
     - `<VSDiskDrive>` is the disk drive where Visual Studio is installed, e.g. `C`
     - `<VSVersion>` is the version of Visual Studio, e.g. `2022`
     - `<Edition>` is the edition of Visual Studio, e.g. `Community`, `Enterprise`
     - `<VCVars>` is the version of Visual C++ toolset, e.g. `14.40`.

     Using the same toolset that has been used for building `libcatboostmodel` library mentioned above is recommended for consistency.
    
### Building

```
cargo build
```

#### GPU support

[CUDA](https://developer.nvidia.com/cuda) support is available for Linux and Windows target platforms.

Inference on CUDA GPUs is currently supported only for models with exclusively numerical features.

It is disabled by default and can be enabled by adding `gpu` to cargo's features list, for example:

```
cargo build --features "gpu"
```

CUDA architectures to generate device code for are specified using [`CMAKE_CUDA_ARCHITECTURES` variable](https://cmake.org/cmake/help/v3.24/variable/CMAKE_CUDA_ARCHITECTURES.html), although the default value is non-standard, [specified in `cuda.cmake`](https://github.com/catboost/catboost/blob/5fb7b9def07f4ea2df6dcc31b5cd1e81a8b00217/cmake/cuda.cmake#L7). The default value is intended to provide broad GPU compatibility and supported only when building with CUDA 11.8.
The most convenient way to override the default value is to use [`CUDAARCHS` environment variable](https://cmake.org/cmake/help/v3.24/envvar/CUDAARCHS.html).


### Basic usage example

1. Add a dependency to your Cargo.toml

Build and link shared library:
```
[dependencies]
catboost = { git = "https://github.com/catboost/catboost" }
```
Build and link static library:
```
[dependencies]
catboost = { git = "https://github.com/catboost/catboost", features = ["static-link"] }
```

2. Now you can apply pretrained model in your code:
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

Tests with GPU can be enabled by adding `gpu` to cargo's features list, for example:

```
cargo test --features "gpu"
```
