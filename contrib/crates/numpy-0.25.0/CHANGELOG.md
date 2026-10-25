# Changelog
- v0.24.0
  - Bump PyO3 dependency to v0.24.0. ([#483](https://github.com/PyO3/rust-numpy/pull/483)
  - Support Python 3.13t "free-threaded" Python. ([#471](https://github.com/PyO3/rust-numpy/pull/471)

- v0.23.0
  - Drop support for PyPy 3.7 and 3.8. ([#470](https://github.com/PyO3/rust-numpy/pull/470))
  - Require `Element: Sync` as part of the free-threading support in PyO3 0.23 ([#469](https://github.com/PyO3/rust-numpy/pull/469))
  - Bump PyO3 dependency to v0.23.0 ([[#457](https://github.com/PyO3/rust-numpy/pull/457)])
    - removed the `gil-refs` feature
    - reintroduced function names without `_bound` suffix + deprecating the old names
    - switched to `IntoPyObject` as trait bound
  - Bump `rustc-hash` dependency to 2.0. ([[#472](https://github.com/PyO3/rust-numpy/pull/472)])

- v0.22.1
  - Fix building on 32-bit Windows. ([#463](https://github.com/PyO3/rust-numpy/pull/463))
  - Add `PyReadwriteArray::make_nonwriteable`. ([#462](https://github.com/PyO3/rust-numpy/pull/462))
  - Implement `From<PyReadWriteArray>` for `PyReadonlyArray`. ([#462](https://github.com/PyO3/rust-numpy/pull/462))

- v0.22.0
  - Bump MSRV to 1.63. ([#450](https://github.com/PyO3/rust-numpy/pull/450))
  - Add `permute` and `transpose` methods for changing the order of axes of a `PyArray`. ([#428](https://github.com/PyO3/rust-numpy/pull/428))
  - Add support for NumPy v2 which had a number of changes to the [C API](https://numpy.org/devdocs/numpy_2_0_migration_guide.html#c-api-changes). ([#442](https://github.com/PyO3/rust-numpy/pull/442))
  - Add support for ndarray 0.16. ([#439](https://github.com/PyO3/rust-numpy/pull/439))
  - Bumped pyo3 dependency to v0.22.0 which required the addition of several new methods to the `Element` trait. ([#435](https://github.com/PyO3/rust-numpy/pull/435))

- v0.21.0
  - Migrate to the new `Bound` API introduced by PyO3 0.21. ([#410](https://github.com/PyO3/rust-numpy/pull/410)) ([#411](https://github.com/PyO3/rust-numpy/pull/411)) ([#412](https://github.com/PyO3/rust-numpy/pull/412)) ([#415](https://github.com/PyO3/rust-numpy/pull/415)) ([#416](https://github.com/PyO3/rust-numpy/pull/416)) ([#418](https://github.com/PyO3/rust-numpy/pull/418)) ([#419](https://github.com/PyO3/rust-numpy/pull/419)) ([#420](https://github.com/PyO3/rust-numpy/pull/420)) ([#421](https://github.com/PyO3/rust-numpy/pull/421)) ([#422](https://github.com/PyO3/rust-numpy/pull/422))
  - Add a `prelude` module to simplify importing method traits required by the `Bound` API. ([#417](https://github.com/PyO3/rust-numpy/pull/417))
  - Extend documentation to cover some more surprising behaviours. ([#405](https://github.com/PyO3/rust-numpy/pull/405)) ([#414](https://github.com/PyO3/rust-numpy/pull/414))

- v0.20.0
  - Increase MSRV to 1.56 released in October 2021 and available in Debain 12, RHEL 9 and Alpine 3.17 following the same change for PyO3. ([#378](https://github.com/PyO3/rust-numpy/pull/378))
  - Add support for ASCII (`PyFixedString<N>`) and Unicode (`PyFixedUnicode<N>`) string arrays, i.e. dtypes `SN` and `UN` where `N` is the number of characters. ([#378](https://github.com/PyO3/rust-numpy/pull/378))
  - Add support for the `bfloat16` dtype by extending the optional integration with the `half` crate. Note that the `bfloat16` dtype is not part of NumPy itself so that usage requires third-party packages like Tensorflow. ([#381](https://github.com/PyO3/rust-numpy/pull/381))
  - Add `PyArrayLike` type which extracts `PyReadonlyArray` if a NumPy array of the correct type is given and attempts a conversion using `numpy.asarray` otherwise. ([#383](https://github.com/PyO3/rust-numpy/pull/383))

- v0.19.0
  - Add `PyUntypedArray` as an untyped base type for `PyArray` which can be used to inspect arguments before more targeted downcasts. This is accompanied by some methods like `dtype` and `shape` moving from `PyArray` to `PyUntypedArray`. They are still accessible though, as `PyArray` dereferences to `PyUntypedArray` via the `Deref` trait. ([#369](https://github.com/PyO3/rust-numpy/pull/369))
  - Drop deprecated `PyArray::from_exact_iter` as it does not provide any benefits over `PyArray::from_iter`. ([#370](https://github.com/PyO3/rust-numpy/pull/370))

- v0.18.0
  - Add conversions from and to datatypes provided by the [`nalgebra` crate](https://nalgebra.org/). ([#347](https://github.com/PyO3/rust-numpy/pull/347))
  - Drop our wrapper for NumPy iterators which were deprecated in v0.16.0 as ndarray's iteration facilities are almost always preferable. ([#324](https://github.com/PyO3/rust-numpy/pull/324))
  - Dynamic borrow checking now uses a capsule-based API and therefore works across multiple extensions using PyO3 and potentially other bindings or languages. ([#361](https://github.com/PyO3/rust-numpy/pull/361))

- v0.17.2
  - Fix unsound aliasing into `Box<[T]>` when converting them into NumPy arrays. ([#351](https://github.com/PyO3/rust-numpy/pull/351))

- v0.17.1
  - Fix use-after-free in `PyArray::resize`, `PyArray::reshape` and `PyArray::reshape_with_order`. ([#341](https://github.com/PyO3/rust-numpy/pull/341))
  - Fix UB in `ToNpyDims::as_dims_ptr` with dimensions of dynamic size (-1). ([#344](https://github.com/PyO3/rust-numpy/pull/344))

- v0.17.0
  - Add dynamic borrow checking to safely construct references into the interior of NumPy arrays. ([#274](https://github.com/PyO3/rust-numpy/pull/274))
    - The deprecated iterator builders `NpySingleIterBuilder::{readonly,readwrite}` and `NpyMultiIterBuilder::add_{readonly,readwrite}` now take referencces to `PyReadonlyArray` and `PyReadwriteArray` instead of consuming them.
    - The destructive `PyArray::resize` method is now unsafe if used without an instance of `PyReadwriteArray`. ([#302](https://github.com/PyO3/rust-numpy/pull/302))
  - Add support for `datetime64` and `timedelta64` element types via the `datetime` module. ([#308](https://github.com/PyO3/rust-numpy/pull/308))
  - Add support for IEEE 754-2008 16-bit floating point numbers via an optional dependency on the `half` crate. ([#314](https://github.com/PyO3/rust-numpy/pull/314))
  - The `inner`, `dot` and `einsum` functions can also return a scalar instead of a zero-dimensional array to match NumPy's types ([#285](https://github.com/PyO3/rust-numpy/pull/285))
  - The `PyArray::resize` function supports n-dimensional contiguous arrays. ([#312](https://github.com/PyO3/rust-numpy/pull/312))
  - Deprecate `PyArray::from_exact_iter` after optimizing `PyArray::from_iter`. ([#292](https://github.com/PyO3/rust-numpy/pull/292))
  - Remove `DimensionalityError` and `TypeError` from the public API as they never used directly. ([#315](https://github.com/PyO3/rust-numpy/pull/315))
  - Remove the deprecated `PyArrayDescr::get_type` which was replaced by `PyArrayDescr::typeobj` in the last cycle. ([#308](https://github.com/PyO3/rust-numpy/pull/308))
  - Fix returning invalid slices from `PyArray::{strides,shape}` for rank zero arrays. ([#303](https://github.com/PyO3/rust-numpy/pull/303))

- v0.16.2
  - Fix build on platforms where `c_char` is `u8` like Linux/AArch64. ([#296](https://github.com/PyO3/rust-numpy/pull/296))

- v0.16.1
  - Fix build when PyO3's `multiple-pymethods` feature is used. ([#288](https://github.com/PyO3/rust-numpy/pull/288))

- v0.16.0
  - Bump PyO3 version to 0.16 ([#259](https://github.com/PyO3/rust-numpy/pull/259))
  - Support object arrays ([#216](https://github.com/PyO3/rust-numpy/pull/216))
  - Support borrowing arrays that are part of other Python objects via `PyArray::borrow_from_array` ([#230](https://github.com/PyO3/rust-numpy/pull/230))
  - Fixed downcasting ignoring element type and dimensionality ([#265](https://github.com/PyO3/rust-numpy/pull/265))
  - `PyArray::new` is now `unsafe`, as it produces uninitialized arrays ([#220](https://github.com/PyO3/rust-numpy/pull/220))
  - `PyArray::iter`, `NpySingleIterBuilder::readwrite` and `NpyMultiIterBuilder::add_readwrite` are now `unsafe`, as they allow aliasing mutable references to be created ([#278/](https://github.com/PyO3/rust-numpy/pull/278))
  - The `npyiter` module is deprecated as rust-ndarray's facilities for iteration are more flexible and performant ([#280](https://github.com/PyO3/rust-numpy/pull/280))
  - `PyArray::from_exact_iter` does not unsoundly trust `ExactSizeIterator::len` any more ([#262](https://github.com/PyO3/rust-numpy/pull/262))
  - `PyArray::as_cell_slice` was removed as it unsoundly interacts with `PyReadonlyArray` allowing safe code to violate aliasing rules ([#260](https://github.com/PyO3/rust-numpy/pull/260))
  - `rayon` feature is now removed, and directly specifying the feature via `ndarray` dependency is recommended ([#250](https://github.com/PyO3/rust-numpy/pull/250))
  - `Element` trait and `PyArrayDescr` changes ([#256](https://github.com/PyO3/rust-numpy/pull/256)):
    - `Element` trait has been simplified to `get_dtype()` and `IS_COPY`
    - New `PyArrayDescr` methods: `of`, `into_dtype_ptr`, `is_equiv_to`
    - Added `numpy::dtype` function
    - `Element` is now implemented for `isize`
    - `c32` / `c64` have been renamed with `Complex32` / `Complex64`
    - `ShapeError` has been split into `TypeError` and `DimensionalityError`
    - `i32`, `i64`, `u32`, `u64` are now guaranteed to map to `np.u?int{32,64}`.
    - Removed `cfg_if` dependency
    - Removed `DataType` enum
  - Added `PyArrayDescr::new` constructor ([#266](https://github.com/PyO3/rust-numpy/pull/266))
  - New `PyArrayDescr` methods ([#266](https://github.com/PyO3/rust-numpy/pull/261)):
    - `num`, `base`, `ndim`, `shape`, `byteorder`, `char`, `kind`, `itemsize`,
      `alignment`, `flags`, `has_object`, `is_aligned_struct`, `names`,
      `get_field`, `has_subarray`, `has_fields`, `is_native_byteorder`
    - Renamed `get_type` to `typeobj`

- v0.15.1
  - Make arrays produced via `IntoPyArray`, i.e. those owning Rust data, writeable ([#235](https://github.com/PyO3/rust-numpy/pull/235))
  - Fix thread-safety in internal API globals ([#222](https://github.com/PyO3/rust-numpy/pull/222))

- v0.15.0
  - [Remove resolver from Cargo.toml](https://github.com/PyO3/rust-numpy/pull/202)
  - [Bump PyO3 to 0.15](https://github.com/PyO3/rust-numpy/pull/212)

- v0.14.1
  - [Fix MSRV support](https://github.com/PyO3/rust-numpy/issues/198)

- v0.14
  - Bump PyO3 to 0.14
  - Fix [conversion bug](https://github.com/PyO3/rust-numpy/pull/194)

- v0.13.2
  - Support ndarray 0.15

- v0.13.1
  - Allow ndarray `>=0.13, < 0.15` to work with Rust 1.41.1.
  - Add inner, dot, and einsum
  - Add PyArray0

- v0.13.0
  - Bump num-complex to 0.3
  - Bump ndarray to 0.14
  - Bump pyo3 to 0.13
  - Drop support for Python 3.5 (as it is now end-of-life).
  - Remove unused `python3` feature

- v0.12.2
  - Pin PyO3 minor versions to 0.12
  - Pin ndarray minor versions to 0.13

- v0.12.1
  - Fix compile error in Rust 1.39

- v0.12.0
  - Introduce `NpySingleIter` and `NpyMultiIter`.
  - Introduce `PyArrayDescr`.

- v0.11.0
  - `PyArray::get` is now unsafe.
  - Introduce `PyArray::get_owned` and `PyReadonlyArray::get`.

- v0.10.0
  - Remove `ErrorKind` and introduce some concrete error types.
  - `PyArray::as_slice`, `PyArray::as_slice_mut`, `PyArray::as_array`, and `PyArray::as_array_mut` is now unsafe.
  - Introduce `PyArray::as_cell_slice`, `PyArray::to_vec`, and `PyArray::to_owned_array`.
  - Rename `TypeNum` trait `Element`, and `NpyDataType` `DataType`.
  - Update PyO3 to 0.11

- v0.9.0
  - Update PyO3 to 0.10.0

- v0.8.0
  - Update PyO3 to 0.9.0
  - Fix SliceBox initialization

- v0.7.0
  - Update PyO3 to 0.8

- v0.6.0
  - Update PyO3 to 0.7
  - Drop Python2 support

- v0.5.0
  - Update PyO3 to 0.6

- v0.4.0
  - Duplicate `PyArrayModule` and import Numpy API automatically
  - Fix memory leak of `IntoPyArray` and add `ToPyArray` crate
  - PyArray has dimension as type parameter. Now it looks like `PyArray<T, D>`
  - Use `ndarray::IntoDimension` to specify dimension
  - Python2 support

- v0.3.1, v0.3.2
  - Just update dependencies

- v0.3.0
  - Breaking Change: Migrated to pyo3 from rust-cpython
  - Some api addition
  - [Static type checking with PhantomData](https://github.com/PyO3/rust-numpy/pull/41)

- v0.2.1
  - NEW: trait `IntoPyErr`, `IntoPyResult` for error translation

- v0.2.0
  - NEW: traits `IntoPyArray`, `ToPyArray`
  - MOD: Interface of `PyArray` creation functions are changed

- v0.1.1
  - Update documents

- v0.1.0
  - First Release
  - Expose unsafe interface of Array and UFunc API
