# ArgMinMax

<div align="center">
  <a href="https://crates.io/crates/argminmax">
    <img src="https://img.shields.io/crates/v/argminmax.svg"/>
  </a>
  <a href="">
    <img src="https://shields.io/badge/license-MIT-blue"/>
  </a>
  <a href="https://docs.rs/argminmax/latest/argminmax/">
    <img src="https://docs.rs/argminmax/badge.svg" alt="rust docs"/>
  </a>
  <a href="https://github.com/jvdd/argminmax/actions">
    <img src="https://github.com/jvdd/argminmax/actions/workflows/ci.yml/badge.svg" alt="Build and test"/>
  </a>
</div>

> Efficient argmin &amp; argmax (in 1 function) with SIMD (SSE, AVX(2), AVX512<sup>1</sup>, NEON<sup>1</sup>) ⚡

<!-- This project uses [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) to compute argmin and argmax in a single function.   -->

🚀 The functions are generic over the type of the array, so it can be used on `&[T]` or `Vec<T>` where `T` can be `f16`<sup>2</sup>, `f32`<sup>2</sup>, `f64`<sup>3</sup>, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.

🤝 The trait is implemented for [`slice`](https://doc.rust-lang.org/std/primitive.slice.html), [`Vec`](https://doc.rust-lang.org/std/vec/struct.Vec.html), 1D [`ndarray::ArrayBase`](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html)<sup>4</sup>, apache [`arrow::PrimitiveArray`](https://docs.rs/arrow/latest/arrow/array/struct.PrimitiveArray.html)<sup>5</sup> and [`arrow2::PrimitiveArray`](https://docs.rs/arrow2/latest/arrow2/array/struct.PrimitiveArray.html)<sup>6</sup>.

⚡ **Runtime CPU feature detection** is used to select the most efficient implementation for the current CPU. This means that the same binary can be used on different CPUs without recompilation. 

👀 The SIMD implementation contains **no if checks**, ensuring that the runtime of the function is independent of the input data its order (best-case = worst-case = average-case).

🪄 **Efficient support for f16 and uints**: through (bijective aka symmetric) bitwise operations, f16 (optional<sup>1</sup>) and uints are converted to ordered integers, allowing to use integer SIMD instructions.

> <i><sup>1</sup> for <code>AVX512</code> and most of <code>NEON</code> you should enable the (default) `"nightly_simd"` feature (requires nightly Rust).</i>  
> <i><sup>2</sup> for <code>f16</code> you should enable the `"half"` feature.</i>  
> <i><sup>3</sup> for <code>f32</code> and <code>f64</code> you should enable the (default) `"float"` feature.</i>  
> <i><sup>4</sup> for <code>ndarray::ArrayBase</code> you should enable the `"ndarray"` feature.</i>  
> <i><sup>5</sup> for <code>arrow::PrimitiveArray</code> you should enable the `"arrow"` feature.</i>  
> <i><sup>6</sup> for <code>arrow2::PrimitiveArray</code> you should enable the `"arrow2"` feature.</i>

## Installing

Add the following to your `Cargo.toml`:

```toml
[dependencies]
argminmax = "0.6.3"
```

## Example usage

```rust
use argminmax::ArgMinMax;  // import trait

let arr: Vec<i32> = (0..200_000).collect();  // create a vector

let (min, max) = arr.argminmax();  // apply extension

println!("min: {}, max: {}", min, max);
println!("arr[min]: {}, arr[max]: {}", arr[min], arr[max]);
```

## Traits

### `ArgMinMax`

Implemented for `ints`, `uints`, and `floats` (if `"float"` feature enabled).

Provides the following functions:
- `argminmax`: returns the index of the minimum and maximum element in the array.
<!-- - `argmin`: returns the index of the minimum element in the array. -->
<!-- - `argmax`: returns the index of the maximum element in the array. -->

When dealing with NaNs, `ArgMinMax` its functions ignore NaNs. For more info see [Limitations](#limitations).

### `NaNArgMinMax`

Implemented for `floats` (if `"float"` feature enabled).

Provides the following functions:
- `nanargminmax`: returns the index of the minimum and maximum element in the array.
<!-- - `nanargmin`: returns the index of the minimum element in the array. -->
<!-- - `nanargmax`: returns the index of the maximum element in the array. -->

When dealing with NaNs, `NaNArgMinMax` its functions return the first NaN its index. For more info see [Limitations](#limitations).

> Tip 💡: if you know that there are no NaNs in your the array, we advise you to use `ArgMinMax` as this should be 5-30% faster than `NaNArgMinMax`.


## Features
- [default] **"nightly_simd"**: enables the use of non-stable SIMD intrinsics (`AVX512` and most of `NEON`), which are only available on nightly Rust.
- [default] **"float"**: support `f32` and `f64` argminmax (uses NaN-handling - [see below](#limitations)).
- **"half"**: support `f16` argminmax (through using the [`half`](https://docs.rs/half/latest/half) crate).
- **"ndarray"**: add `ArgMinMax` trait to [`ndarray`](https://docs.rs/ndarray/latest/ndarray) its `Array1` & `ArrayView1`.
- **"arrow"**: add `ArgMinMax` trait to [`arrow`](https://docs.rs/arrow/latest/arrow) its `PrimitiveArray`.

## Benchmarks

Benchmarks on my laptop *(AMD Ryzen 7 4800U, 1.8 GHz, 16GB RAM)* using [criterion](https://github.com/bheisler/criterion.rs) show that the function is 3-20x faster than the scalar implementation (depending of data type).

See `/benches/results`.

<!-- *For example, finding the argmin & argmax in an array of 10,000,000  random `f32` elements is 3.5x faster than the scalar implementation (taking 2.4ms vs 8.5ms).* -->

Run the benchmarks yourself with the following command:
```bash
cargo bench --quiet --message-format=short --features half | grep "time:"
```

## Tests

To run the tests use the following command:
```bash
cargo test --message-format=short --all-features
```

## Limitations

The library handles NaNs! 🚀 

<!-- For NaN-handling there are two variants:
- **Ignore NaN**: NaNs are ignored and the index of the highest / lowest non-NaN value is returned.
- **Return NaN**: the first NaN value is returned. -->

Some (minor) limitations:
- `ArgMinMax` its functions ignores NaN values.
  - ❗ When the array contains exclusively NaNs and/or infinities unexpected behaviour can occur (index 0 is returned).
- `NaNArgMinMax` its functions returns the first NaN its index (if any present).
  - ❗ When multiple bit-representations for NaNs are used, no guarantee is made that the first NaN is returned.

---

## Acknowledgements

Some parts of this library are inspired by the great work of [minimalrust](https://github.com/minimalrust)'s [argmm](https://github.com/minimalrust/argmm) project.
