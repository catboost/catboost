# SIMD JSON for Rust &emsp; [![Build Status]][simd-json.rs] [![Build Status ARM]][drone.io] [![Quality]][simd-json.rs]  [![Latest Version]][crates.io] [![Code Coverage]][coveralls]

[Build Status ARM]: https://cloud.drone.io/api/badges/simd-lite/simd-json/status.svg
[drone.io]: https://cloud.drone.io/simd-lite/simd-json
[Build Status]: https://github.com/simd-lite/simd-json/workflows/Tests/badge.svg
[Quality]: https://github.com/simd-lite/simd-json/workflows/Quality/badge.svg
[simd-json.rs]: https://simd-json.rs
[Latest Version]: https://img.shields.io/crates/v/simd-json.svg
[crates.io]: https://crates.io/crates/simd-json
[Code Coverage]: https://coveralls.io/repos/github/simd-lite/simd-json/badge.svg?branch=main
[coveralls]: https://coveralls.io/github/simd-lite/simd-json?branch=main

**Rust port of extremely fast [simdjson](https://github.com/lemire/simdjson) JSON parser with [Serde][serde] compatibility.**

---

simd-json is a Rust port of the [simdjson c++ library](https://simdjson.org/).
It follows most of the design closely with a few exceptions to make it better
fit into the Rust ecosystem.

## Goals

The goal of the Rust port of simdjson is not to create a one-to-one
copy, but to integrate the principles of the C++ library into
a Rust library that plays well with the Rust ecosystem. As such
we provide both compatibility with Serde as well as parsing to a
DOM to manipulate data.

## Performance

As a rule of thumb this library tries to get as close as possible
to the performance of the C++ implementation (currently tracking 0.2.x, work in progress).
However, in some design decisions—such as parsing to a DOM or a tape—ergonomics is prioritized over
performance. In other places Rust makes it harder to achieve the same level of performance.

To take advantage of this library your system needs to support SIMD instructions. On `x86`, it will
select the best available supported instruction set (`avx2` or `sse4.2`) when the `runtime-detection` feature
is enabled (default). On `aarch64` this library uses the `NEON` instruction set. On `wasm` this library uses 
the `simd128` instruction set when available. When no supported SIMD instructions are found, this library will use a
fallback implementation, but this is significantly slower.

### Allocator
For best performance, we highly suggest using [mimalloc](https://crates.io/crates/mimalloc) or [jemalloc](https://crates.io/crates/jemalloc) instead of the system allocator used by
default. Another recent allocator that works well (but we have yet to test it in production) is [snmalloc](https://github.com/microsoft/snmalloc).


## Safety

`simd-json` uses **a lot** of unsafe code.

There are a few reasons for this:

* SIMD intrinsics are inherently unsafe. These uses of unsafe are inescapable in a library such as `simd-json`.
* We work around some performance bottlenecks imposed by safe rust. These are avoidable, but at a performance cost.
  This is a more considered path in `simd-json`.


`simd-json` goes through extra scrutiny for unsafe code. These steps are:

* Unit tests - to test 'the obvious' cases, edge cases, and regression cases
* Structural constructive property based testing - We generate random valid JSON objects to exercise the full `simd-json`
  codebase stochastically. Floats are currently excluded since slightly different parsing algorithms lead to slightly
  different results here. In short "is simd-json correct".
* Data-oriented property-based testing of string-like data - to assert that sequences of legal printable characters
  don't panic or crash the parser (they might and often error so - they are not valid JSON!)
* Destructive Property based testing - make sure that no illegal byte sequences crash the parser in any way
* Fuzzing - fuzz based on upstream & jsonorg simd pass/fail cases

This doesn't ensure complete safety nor is at a bulletproof guarantee, but it does go a long way
to assert that the library is of high production quality and fit for purpose for practical industrial applications.

## Features
Various features can be enabled or disabled to tweak various parts of this library. Any features not mentioned here are
for internal configuration and testing.

### `runtime-detection` (default)

This feature allows selecting the optimal algorithm based on available features during runtime. It has no effect on
non-`x86` platforms. When neither `AVX2` nor `SSE4.2` is supported, it will fall back to a native Rust implementation.

Disabling this feature (with `default-features = false`) **and** setting `RUSTFLAGS="-C target-cpu=native` will result
in better performance but the resulting binary will not be portable across `x86` processors.

### `serde_impl` (default)

Enable [Serde](https://serde.rs) support. This consist of implementing `serde::Serializer` and `serde::Deserializer`,
allowing types that implement `serde::Serialize`/`serde::Deserialize` to be constructed/serialized to 
`BorrowedValue`/`OwnedValue`.
In addition, this provides the same convenience functions that [`serde_json`](https://docs.rs/serde_json/latest/serde_json/) provides.

Disabling this feature (with `default-features = false`) will remove `serde` and `serde_json` from the dependencies.

### `swar-number-parsing` (default)
Enables a parsing method that will parse 8 digits at a time for floats. This is a common pattern but comes at a slight
performance hit if most of the float have less than 8 digits.

### `known-key`

The `known-key` feature changes the hash mechanism for the DOM representation of the underlying JSON object from
`ahash` to `fxhash`. The `ahash` hasher is faster at hashing and provides protection against DOS attacks by forcing
multiple keys into a single hashing bucket. The `fxhash` hasher allows for repeatable hashing results,
which in turn allows memoizing hashes for well known keys and saving time on lookups. In workloads that are heavy on
accessing some well-known keys, this can be a performance advantage.

The `known-key` feature is optional and disabled by default and should be explicitly configured.

### `value-no-dup-keys`

**This flag has no effect on simd-json itself but purely affects the `Value` structs.**

The `value-no-dup-keys` feature flag enables stricter behavior for objects when deserializing into a `Value`. When
enabled, the Value deserializer will remove duplicate keys in a JSON object and only keep the last one. If not set
duplicate keys are considered undefined behavior and Value will not make guarantees on its behavior.

### `big-int-as-float`

The `big-int-as-float` feature flag treats very large integers that won't fit into u64 as f64 floats. This prevents
parsing errors if the JSON you are parsing contains very large integers. Keep in mind that f64 loses some precision when
representing very large numbers.

### `128bit`

Add support for parsing and serializing 128-bit integers. This feature is disabled by default because such large numbers
are rare in the wild and adding the support incurs a performance penalty.

### `beef`

**Enabling this feature can break dependencies in your dependency tree that are using `simd-json`.**

Replace [`std::borrow::Cow`](https://doc.rust-lang.org/std/borrow/enum.Cow.html) with
[`beef::lean::Cow`][beef] This feature is disabled by default, because
it is a breaking change in the API. 

### `ordered-float`

By default the representation of `Floats` used in `borrowed::Value ` and `owned::Value` is simply a value of `f64`. 
This however has the normally-not-a-big-deal side effect of _not_ having these `Value` types be `std::cmp::Eq`. This does,
however, introduce some incompatibilities when offering `simd-json` as a quasi-drop-in replacement for `serde-json`.

So, this feature changes the internal representation of `Floats` to be an `f64` _wrapped by [an Eq-compatible adapter](https://docs.rs/ordered-float/latest/ordered_float/)_.

This probably carries with it some small performance trade-offs, hence its enablement by feature rather than by default.

### `portable`

**Currently disabled**

An highly experimental implementation of the algorithm using `std::simd` and up to 512 byte wide registers.


## Usage

simd-json offers three main entry points for usage:

### Values API

The values API is a set of optimized DOM objects that allow parsed
JSON to JSON data that has no known variable structure. `simd-json`
has two versions of this:

**Borrowed Values**

```rust
use simd_json;
let mut d = br#"{"some": ["key", "value", 2]}"#.to_vec();
let v: simd_json::BorrowedValue = simd_json::to_borrowed_value(&mut d).unwrap();
```

**Owned Values**

```rust
use simd_json;
let mut d = br#"{"some": ["key", "value", 2]}"#.to_vec();
let v: simd_json::OwnedValue = simd_json::to_owned_value(&mut d).unwrap();
```

### Serde Compatible API

```rust ignore
use simd_json;
use serde_json::Value;

let mut d = br#"{"some": ["key", "value", 2]}"#.to_vec();
let v: Value = simd_json::serde::from_slice(&mut d).unwrap();
```

### Tape API

```rust
use simd_json;

let mut d = br#"{"the_answer": 42}"#.to_vec();
let tape = simd_json::to_tape(&mut d).unwrap();
let value = tape.as_value();
// try_get treats value like an object, returns Ok(Some(_)) because the key is found
assert!(value.try_get("the_answer").unwrap().unwrap() == 42);
// returns Ok(None) because the key is not found but value is an object
assert!(value.try_get("does_not_exist").unwrap() == None);
// try_get_idx treats value like an array, returns Err(_) because value is not an array
assert!(value.try_get_idx(0).is_err());
```

## Other interesting things

There are also bindings for upstream `simdjson` available [here](https://github.com/SunDoge/simdjson-rust)

## License

simd-json itself is licensed under either of

* [Apache License, Version 2.0, (LICENSE-APACHE)](http://www.apache.org/licenses/LICENSE-2.0)
* [MIT license (LICENSE-MIT)](http://opensource.org/licenses/MIT)

at your option.

However it ports a lot of code from [simdjson](https://github.com/lemire/simdjson) so their work and copyright on that should also be respected.

The [Serde][serde] integration is based on `serde-json` so their copyright should as well be respected.

[serde]: https://serde.rs
[beef]: https://docs.rs/beef/latest/beef/lean/type.Cow.html

### All Thanks To Our Contributors:
<a href="https://github.com/simd-lite/simd-json/graphs/contributors">
  <img alt="GitHub profile pictures of all contributors to simd-json" src="https://contrib.rocks/image?repo=simd-lite/simd-json" />
</a>
