rust-numpy
===========
[![Actions Status](https://github.com/PyO3/rust-numpy/workflows/CI/badge.svg)](https://github.com/PyO3/rust-numpy/actions)
[![Crate](https://img.shields.io/crates/v/numpy.svg)](https://crates.io/crates/numpy)
[![Minimum rustc 1.63](https://img.shields.io/badge/rustc-1.63+-blue.svg)](https://rust-lang.github.io/rfcs/2495-min-rust-version.html)
[![Documentation](https://docs.rs/numpy/badge.svg)](https://docs.rs/numpy)
[![codecov](https://codecov.io/gh/PyO3/rust-numpy/branch/main/graph/badge.svg)](https://codecov.io/gh/PyO3/rust-numpy)

Rust bindings for the NumPy C-API.

## API documentation
- [Latest release](https://docs.rs/numpy)
- [Current main](https://pyo3.github.io/rust-numpy)

## Requirements
- Rust >= 1.63.0
  - Basically, our MSRV follows the one of [PyO3](https://github.com/PyO3/pyo3)
- Python >= 3.7
  - Python 3.6 support was dropped from 0.16
- Some Rust libraries
  - [ndarray](https://github.com/rust-ndarray/ndarray) for Rust-side matrix library
  - [PyO3](https://github.com/PyO3/pyo3) for Python bindings
  - And more (see [Cargo.toml](Cargo.toml))
- [numpy](https://numpy.org/) installed in your Python environments (e.g., via `pip install numpy`)
  - We recommend `numpy >= 1.16.0`, though older versions may work

## Example

### Write a Python module in Rust

Please see the [simple](examples/simple) example for how to get started.

There are also examples using [ndarray-linalg](examples/linalg) and [rayon](examples/parallel).

```toml
[lib]
name = "rust_ext"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"] }
numpy = "0.22"
```

```rust
use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyArrayMethods};
use pyo3::{pymodule, types::PyModule, PyResult, Python, Bound};

#[pymodule]
fn rust_ext<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    // example using immutable borrows producing a new array
    fn axpy(a: f64, x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> ArrayD<f64> {
        a * &x + &y
    }

    // example using a mutable borrow to modify an array in-place
    fn mult(a: f64, mut x: ArrayViewMutD<'_, f64>) {
        x *= a;
    }

    // wrapper of `axpy`
    #[pyfn(m)]
    #[pyo3(name = "axpy")]
    fn axpy_py<'py>(
        py: Python<'py>,
        a: f64,
        x: PyReadonlyArrayDyn<'py, f64>,
        y: PyReadonlyArrayDyn<'py, f64>,
    ) -> Bound<'py, PyArrayDyn<f64>> {
        let x = x.as_array();
        let y = y.as_array();
        let z = axpy(a, x, y);
        z.into_pyarray(py)
    }

    // wrapper of `mult`
    #[pyfn(m)]
    #[pyo3(name = "mult")]
    fn mult_py<'py>(a: f64, x: &Bound<'py, PyArrayDyn<f64>>) {
        let x = unsafe { x.as_array_mut() };
        mult(a, x);
    }

    Ok(())
}
```

### Execute a Python program from Rust and get results

``` toml
[package]
name = "numpy-test"

[dependencies]
pyo3 = { version = "0.22", features = ["auto-initialize"] }
numpy = "0.22"
```

```rust
use numpy::{PyArray1, PyArrayMethods};
use pyo3::{types::{IntoPyDict, PyAnyMethods}, PyResult, Python, ffi::c_str};

fn main() -> PyResult<()> {
    Python::with_gil(|py| {
        let np = py.import("numpy")?;
        let locals = [("np", np)].into_py_dict(py)?;

        let pyarray = py
            .eval(c_str!("np.absolute(np.array([-1, -2, -3], dtype='int32'))"), Some(&locals), None)?
            .downcast_into::<PyArray1<i32>>()?;

        let readonly = pyarray.readonly();
        let slice = readonly.as_slice()?;
        assert_eq!(slice, &[1, 2, 3]);

        Ok(())
    })
}
```

## Dependency on ndarray

This crate uses types from `ndarray` in its public API. `ndarray` is re-exported
in the crate root so that you do not need to specify it as a direct dependency.

Furthermore, this crate is compatible with multiple versions of `ndarray` and therefore depends
on a range of semver-incompatible versions, currently `>= 0.15, < 0.17`. Cargo does not
automatically choose a single version of `ndarray` by itself if you depend directly or indirectly
on anything but that exact range. It can therefore be necessary to manually unify these dependencies.

For example, if you specify the following dependencies

```toml
numpy = "0.22"
ndarray = "0.15"
```

this will currently depend on both version `0.15.6` and `0.16.1` of `ndarray` by default
even though `0.15.6` is within the range `>= 0.15, < 0.17`. To fix this, you can run

```sh
cargo update --package ndarray:0.16.1 --precise 0.15.6
```

to achieve a single dependency on version `0.15.6` of `ndarray`.

## Contributing

We welcome [issues](https://github.com/PyO3/rust-numpy/issues)
and [pull requests](https://github.com/PyO3/rust-numpy/pulls).

PyO3's [Contributing.md](https://github.com/PyO3/pyo3/blob/main/Contributing.md)
is a nice guide for starting.

Also, we have a [Gitter](https://gitter.im/PyO3/Lobby) channel for communicating.
