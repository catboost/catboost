# Contributing to argminmax

Welcome! We're happy to have you here. Thank you in advance for your contribution to argminmax.

## The basics

argminmax welcomes contributions in the form of Pull Requests. For small changes (e.g., bug fixes), feel free to submit a PR. For larger changes (e.g., new functionality, major refactoring), consider submitting an [Issue](https://github.com/jvdd/argminmax/issues) outlining your proposed change.

### Prerequisites

argminmax is written in Rust. You'll need to install the [Rust toolchain](https://www.rust-lang.org/tools/install) for development.  

This project uses the nightly version of Rust. You can install it with:

```bash
rustup install nightly
```

and then set it as the default toolchain with:

```bash
rustup default nightly
```

### argminmax 

The structure of the argminmax project is as follows:

```bash
argminmax
├── Cargo.toml
├── README.md
├── src
│   ├── lib.rs     # ArgMinMax trait implementation
│   ├── scalar     # Scalar implementation
│   ├── simd       # SIMD implementation
├── benches        # Benchmarks
├── tests          # Integration tests
├── dev-utils      # Helper functions (for testing & benchmarking)
```

The Rust code is located in the `src` directory. The `lib.rs` file contains the trait implementation. The `scalar` and `simd` directories contain the scalar and SIMD implementations respectively.

The `simd` directory contains a `generic.rs` file that contains the `SIMD` trait - can be seen as an interface that all the SIMD instruction sets x data types must implement. The implementations of this `SIMD` trait are located in the `simd_xxx.rs` files (e.g., `simd_f32.rs`).

### Testing

Unit tests are located at the bottom of (almost) every Rust file. The integration tests are located in the `tests` directory.

They can be run with:
```bash
cargo test --all-features
```

### Benchmarking

The benchmarks are located in the `benches` directory - they are written using [criterion](https://docs.rs/criterion/latest/criterion).

To run the benchmarks, use the following command:

```bash
cargo bench --quiet --message-format=short --features half | grep "time:"
```

### Formatting 

To format the Rust code, run the following command:
```sh
cargo fmt
```

---

## Improving the performance

When a PR is submitted that improves the performance of the library, we would highly appreciate if the PR also includes a (verifiable) benchmark that shows the improvement.
