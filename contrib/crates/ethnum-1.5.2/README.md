# `ethnum`

This crate provides implementations for 256-bit integers, the primitive integer
type in Ethereum. This implementation is meant to be as close as possible to
Rust integer primitives, implementing the same methods and traits.

## Usage

Add this to your `Cargo.toml`:

```toml
ethnum = "1"
```

The API follows the Rust `{i,u}N` primitive types as close as possible.

### Macros

This crate provides `const fn` based macros for 256-bit integer literals. This
allows you to specify 256-bit signed and unsigned integer literals (that can,
for example, be used as `const`s) that are larger than the largest native
integer literal (`i128::MIN` and `i128::MAX` for signed integers and `u128::MAX`
for unsigned integers):

```rust
int!("-57896044618658097711785492504343953926634992332820282019728792003956564819968");
int!("57896044618658097711785492504343953926634992332820282019728792003956564819967");
uint!("115792089237316195423570985008687907853269984665640564039457584007913129639935");
```

Note that these literals support prefixes (`0b` for binary, `0o` for octal, and
`0x` for hexadecimal) as well as `_` and whitespace separators:

```rust
int!("-0b1010101010101010101010101010101010101010101010101010101010101010
         0101010101010101010101010101010101010101010101010101010101010101");
int!("0o 0123 4567");
uint!("0xffff_ffff");
```

## Features

### `macros`

The `macros` feature used to enable 256-bit integer literals via procedural
macros. However, this crate now implements these macros with `const fn`, so the
feature is now deprecated and the macros are now always available. The feature
is still around to not break semantic versioning, but will be removed in a
version `2`.

### `serde`

The `serde` feature adds support for `serde` serialization and deserialization.
By default, the 256-bit integer types are serialized as prefixed hexadecimal
strings. Various serialization helpers are also provided for more fine-grained
control over how serialization is performed.

## Intrinsics

The 256-bit integers uses intrinsics based on two implementations:

### Native Rust Implementation

The integer intrinsics are implemented using standard Rust. The more complicated
operations such as multiplication and division are ported from C compiler
intrinsics for implementing equivalent 128-bit operations on 64-bit systems (or
64-bit operations on 32-bit systems). In general, these are ported from the
Clang `compiler-rt` support routines.

This is the default implementation used by the crate, and in general is quite
well optimized. When using native the implementation, there are no additional
dependencies for this crate.

### LLVM Generated Implementation

Alternatively, `ethnum` can use LLVM-generated intrinsics for base 256-bit
integer operations. This takes advantage of the fact that LLVM IR supports
arbitrarily sized integer operations (such as `@llvm.uadd.with.overflow.i256`
for overflowing unsigned addition). This will produce more optimized assembly
for things like addition and multiplication.

However, there are a couple downsides to using LLVM-generated intrinsics. First
of all, Clang is required in order to compile the LLVM IR. Additionally, Rust
usually optimizes when compiling and linking Rust code (and not externally
linked code), this means that these intrinsics cannot be inlined adding an extra
function call overhead in some cases which make it perform worse than the native
Rust implementation despite having more optimized assembly. Luckily, Rust
currently has support for linker plugin LTO to enable optimizations during the
link step, enabling optimizations with Clang-compiled LLVM IR.

In order to use LLVM-generated intrinsics, enable the `llvm-intrinsics` feature:

```toml
ethnum = { version = "1", features = ["llvm-intrinsics"] }
```

And, genererally it is a good idea to compile with `linker-plugin-lto` enabled
in order to actually take advantage of the the optimized assembly:

```sh
RUSTFLAGS="-Clinker-plugin-lto -Clinker=clang -Clink-arg=-fuse-ld=lld" cargo build
```

Note that **the `clang` version must match the `rustc` LLVM version**. If not,
it is possible to encounter errors when running the `ethnum-intrinsics` build
script. You can verify the LLVM version used by `rustc` with:

```sh
rustc --version --verbose | grep LLVM
```

In particular, this affects macOS which ships its own `clang` binary. The
`ethnum-intrinsics` build script accepts a `CLANG` environment variable to
specity a specific `clang` executable path to use. Using the major LLVM version
from the command above:

```
brew install llvm@${LLVM_VERSION}
CLANG=/opt/homebrew/opt/llvm@${LLVM_VERSION}/bin/clang cargo build
```

### API Stability

The instinsics are exported under `ethnum::intrinsics`. That being said, be
careful when using these intrinsics directly. Semantic versioning API
compatibility is **not guaranteed** for any of these intrinsics.

If you do you use these in your projects, it is recommended to use strict
versioning:

```toml
[dependencies]
ethnum = "=x.y.z"
```

This will ensure commands like `cargo update` won't change the version of the
`ethnum` dependency.

## Benchmarking

The `ethnum-bench` crate implements `criterion` benchmarks for performance of
integer intrinsics:

```sh
cargo bench -p ethnum-bench
RUSTFLAGS="-Clinker-plugin-lto -Clinker=clang -Clink-arg=-fuse-ld=lld" cargo bench -p ethnum-bench --features llvm-intrinsics
```

## Fuzzing

The `ethnum-fuzz` crate implements an AFL fuzzing target (as well as some
utilities for working with `cargo afl`). Internally, it converts the signed
256-bit integer types to `num::BigInt` and uses its operation implementations as
a reference.

In order to start fuzzing:

```sh
cargo install --force cargo-afl
cargo run -p ethnum-fuzz --bin init target/fuzz
cargo afl build -p ethnum-fuzz --bin fuzz
cargo afl fuzz -i target/fuzz/in -o target/fuzz/out target/debug/fuzz
```

In order to replay crashes:

```sh
cargo run -p ethnum-fuzz --bin dump target/fuzz/out/default/crashes/FILE
```
