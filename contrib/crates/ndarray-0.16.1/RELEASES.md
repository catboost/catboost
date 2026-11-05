Version 0.16.1 (2024-08-14)
===========================

- Refactor and simplify BLAS gemm call further by [@bluss](https://github.com/bluss) [#1421](https://github.com/rust-ndarray/ndarray/pull/1421)
- Fix infinite recursion and off-by-one error in triu/tril by [@akern40](https://github.com/akern40) [#1418](https://github.com/rust-ndarray/ndarray/pull/1418)
- Fix using BLAS for all compatible cases of memory layout by [@bluss](https://github.com/bluss) [#1419](https://github.com/rust-ndarray/ndarray/pull/1419)
- Use PR check instead of Merge Queue, and check rustdoc by [@bluss](https://github.com/bluss) [#1420](https://github.com/rust-ndarray/ndarray/pull/1420)
- Make iterators covariant in element type by [@bluss](https://github.com/bluss) [#1417](https://github.com/rust-ndarray/ndarray/pull/1417)

Version 0.16.0 (2024-08-03)
===========================

Featured Changes
----------------

- Better shape: Deprecate reshape, into_shape by [@bluss](https://github.com/bluss) [#1310](https://github.com/rust-ndarray/ndarray/pull/1310)<br>
  `.into_shape()` **is now deprecated**.
  Use `.into_shape_with_order()` or `.to_shape()` instead, which don't have `into_shape`'s drawbacks.

New Features and Improvements
-----------------------------

- Check for aliasing in `RawViewMut::from_shape_ptr` with a debug assertion by [@bluss](https://github.com/bluss) [#1413](https://github.com/rust-ndarray/ndarray/pull/1413)
- Allow aliasing in ArrayView::from_shape by [@bluss](https://github.com/bluss) [#1410](https://github.com/rust-ndarray/ndarray/pull/1410)
- Remove deprecations from 0.15.x by [@bluss](https://github.com/bluss) [#1409](https://github.com/rust-ndarray/ndarray/pull/1409)
- Make `CowArray` an owned storage array, require Clone bound for `into_shared` by [@jturner314](https://github.com/jturner314) [#1028](https://github.com/rust-ndarray/ndarray/pull/1028)
- Change `NdProducer::Dim` of `axis_windows()` to `Ix1` by [@jonasBoss](https://github.com/jonasBoss) [#1305](https://github.com/rust-ndarray/ndarray/pull/1305)
- Add `squeeze()` to dynamic dimension arrays by [@barakugav](https://github.com/barakugav) [#1396](https://github.com/rust-ndarray/ndarray/pull/1396)
- Add `flatten`, `flatten_with_order` and `into_flat` to arrays by [@barakugav](https://github.com/barakugav) [#1397](https://github.com/rust-ndarray/ndarray/pull/1397)
- Make compatible with thumbv6m-none-eabi by [@BjornTheProgrammer](https://github.com/BjornTheProgrammer) [#1384](https://github.com/rust-ndarray/ndarray/pull/1384)
- `is_unique` for `ArcArray` by [@daniellga](https://github.com/daniellga) [#1399](https://github.com/rust-ndarray/ndarray/pull/1399)
- Add `triu` and `tril` methods directly to ArrayBase by [@akern40](https://github.com/akern40) [#1386](https://github.com/rust-ndarray/ndarray/pull/1386)
- Fix styling of the BLAS integration heading. by [@adamreichold](https://github.com/adamreichold) [#1390](https://github.com/rust-ndarray/ndarray/pull/1390)
- Implement `product_axis` by [@akern40](https://github.com/akern40) [#1387](https://github.com/rust-ndarray/ndarray/pull/1387)
- Add reserve method for owned arrays by [@ssande7](https://github.com/ssande7) [#1268](https://github.com/rust-ndarray/ndarray/pull/1268)
- Use inline on spit_at and smaller methods by [@bluss](https://github.com/bluss) [#1381](https://github.com/rust-ndarray/ndarray/pull/1381)
- Update to Approx 0.5 by [@bluss](https://github.com/bluss) [#1380](https://github.com/rust-ndarray/ndarray/pull/1380)
- Add .into_raw_vec_with_offset() and deprecate .into_raw_vec() by [@bluss](https://github.com/bluss) [#1379](https://github.com/rust-ndarray/ndarray/pull/1379)
- Add additional array -> array view conversions by [@bluss](https://github.com/bluss) [#1130](https://github.com/rust-ndarray/ndarray/pull/1130)
- implement DoubleEndedIterator for 1d `LanesIter` by [@Muthsera](https://github.com/Muthsera) [#1237](https://github.com/rust-ndarray/ndarray/pull/1237)
- Add Zip::any by [@nilgoyette](https://github.com/nilgoyette) [#1228](https://github.com/rust-ndarray/ndarray/pull/1228)
- Make the aview0, aview1, and aview2 free functions be const fns by [@jturner314](https://github.com/jturner314) [#1132](https://github.com/rust-ndarray/ndarray/pull/1132)
- Add missing safety checks to `From<&[[A; N]]> for ArrayView` and `From<&mut [[A; N]]> for ArrayViewMut` by [@jturner314](https://github.com/jturner314) [#1131](https://github.com/rust-ndarray/ndarray/pull/1131)
- derived Debug for Iter and IterMut by [@biskwikman](https://github.com/biskwikman) [#1353](https://github.com/rust-ndarray/ndarray/pull/1353)
- Fix Miri errors for WindowsIter and ExactChunksIter/Mut by [@jturner314](https://github.com/jturner314) [#1142](https://github.com/rust-ndarray/ndarray/pull/1142)
- Fix Miri failure with -Zmiri-tag-raw-pointers by [@jturner314](https://github.com/jturner314) [#1138](https://github.com/rust-ndarray/ndarray/pull/1138)
- Track-caller panics by [@xd009642](https://github.com/xd009642) [#975](https://github.com/rust-ndarray/ndarray/pull/975)
- Add slice_axis_move method by [@jturner314](https://github.com/jturner314) [#1211](https://github.com/rust-ndarray/ndarray/pull/1211)
- iterators: Re-export IntoIter by [@bluss](https://github.com/bluss) [#1370](https://github.com/rust-ndarray/ndarray/pull/1370)
- Fix unsafe blocks in `s![]` macro by [@jturner314](https://github.com/jturner314) [#1196](https://github.com/rust-ndarray/ndarray/pull/1196)
- Fix comparison with NumPy of slicing with negative step by [@venkat0791](https://github.com/venkat0791) [#1319](https://github.com/rust-ndarray/ndarray/pull/1319)
- Updated Windows `base` Computations to be Safer by [@LazaroHurtado](https://github.com/LazaroHurtado) [#1297](https://github.com/rust-ndarray/ndarray/pull/1297)
- Update README-quick-start.md by [@fumseckk](https://github.com/fumseckk) [#1246](https://github.com/rust-ndarray/ndarray/pull/1246)
- Added stride support to `Windows` by [@LazaroHurtado](https://github.com/LazaroHurtado) [#1249](https://github.com/rust-ndarray/ndarray/pull/1249)
- Added select example to numpy user docs by [@WillAyd](https://github.com/WillAyd) [#1294](https://github.com/rust-ndarray/ndarray/pull/1294)
- Add both approx features to the readme by [@nilgoyette](https://github.com/nilgoyette) [#1289](https://github.com/rust-ndarray/ndarray/pull/1289)
- Add NumPy examples combining slicing and assignment by [@jturner314](https://github.com/jturner314) [#1210](https://github.com/rust-ndarray/ndarray/pull/1210)
- Fix contig check for single element arrays by [@bluss](https://github.com/bluss) [#1362](https://github.com/rust-ndarray/ndarray/pull/1362)
- Export Linspace and Logspace iterators by [@johann-cm](https://github.com/johann-cm) [#1348](https://github.com/rust-ndarray/ndarray/pull/1348)
- Use `clone_from()` in two places by [@ChayimFriedman2](https://github.com/ChayimFriedman2) [#1347](https://github.com/rust-ndarray/ndarray/pull/1347)
- Update README-quick-start.md by [@joelchen](https://github.com/joelchen) [#1344](https://github.com/rust-ndarray/ndarray/pull/1344)
- Provide element-wise math functions for floats by [@KmolYuan](https://github.com/KmolYuan) [#1042](https://github.com/rust-ndarray/ndarray/pull/1042)
- Improve example in doc for columns method by [@gkobeaga](https://github.com/gkobeaga) [#1221](https://github.com/rust-ndarray/ndarray/pull/1221)
- Fix description of stack! in quick start by [@jturner314](https://github.com/jturner314) [#1156](https://github.com/rust-ndarray/ndarray/pull/1156)

Tests, CI and Maintainer tasks
------------------------------

- CI: require rustfmt, nostd by [@bluss](https://github.com/bluss) [#1411](https://github.com/rust-ndarray/ndarray/pull/1411)
- Prepare changelog for 0.16.0 by [@bluss](https://github.com/bluss) [#1401](https://github.com/rust-ndarray/ndarray/pull/1401)
- Organize dependencies with workspace = true (cont.) by [@bluss](https://github.com/bluss) [#1407](https://github.com/rust-ndarray/ndarray/pull/1407)
- Update to use dep: for features by [@bluss](https://github.com/bluss) [#1406](https://github.com/rust-ndarray/ndarray/pull/1406)
- Organize the workspace of test crates a bit better by [@bluss](https://github.com/bluss) [#1405](https://github.com/rust-ndarray/ndarray/pull/1405)
- Add rustfmt commit to ignored revisions for git blame by [@lucascolley](https://github.com/lucascolley) [#1376](https://github.com/rust-ndarray/ndarray/pull/1376)
- The minimum amount of work required to fix our CI by [@adamreichold](https://github.com/adamreichold) [#1388](https://github.com/rust-ndarray/ndarray/pull/1388)
- Fixed broke continuous integration badge by [@juhotuho10](https://github.com/juhotuho10) [#1382](https://github.com/rust-ndarray/ndarray/pull/1382)
- Use mold linker to speed up ci by [@bluss](https://github.com/bluss) [#1378](https://github.com/rust-ndarray/ndarray/pull/1378)
- Add rustformat config and CI by [@bluss](https://github.com/bluss) [#1375](https://github.com/rust-ndarray/ndarray/pull/1375)
- Add docs to CI by [@jturner314](https://github.com/jturner314) [#925](https://github.com/rust-ndarray/ndarray/pull/925)
- Test using cargo-careful by [@bluss](https://github.com/bluss) [#1371](https://github.com/rust-ndarray/ndarray/pull/1371)
- Further ci updates - numeric tests, and run all tests on PRs by [@bluss](https://github.com/bluss) [#1369](https://github.com/rust-ndarray/ndarray/pull/1369)
- Setup ci so that most checks run in merge queue only by [@bluss](https://github.com/bluss) [#1368](https://github.com/rust-ndarray/ndarray/pull/1368)
- Use merge queue by [@bluss](https://github.com/bluss) [#1367](https://github.com/rust-ndarray/ndarray/pull/1367)
- Try to make the master branch shipshape by [@adamreichold](https://github.com/adamreichold) [#1286](https://github.com/rust-ndarray/ndarray/pull/1286)
- Update ci - run cross tests only on master by [@bluss](https://github.com/bluss) [#1366](https://github.com/rust-ndarray/ndarray/pull/1366)
- ndarray_for_numpy_users some example to code not pointed out to clippy by [@higumachan](https://github.com/higumachan) [#1360](https://github.com/rust-ndarray/ndarray/pull/1360)
- Fix minimum rust version mismatch in lib.rs by [@HoKim98](https://github.com/HoKim98) [#1352](https://github.com/rust-ndarray/ndarray/pull/1352)
- Fix MSRV build by pinning crossbeam crates. by [@adamreichold](https://github.com/adamreichold) [#1345](https://github.com/rust-ndarray/ndarray/pull/1345)
- Fix new rustc lints to make the CI pass. by [@adamreichold](https://github.com/adamreichold) [#1337](https://github.com/rust-ndarray/ndarray/pull/1337)
- Make Clippy happy and fix MSRV build by [@adamreichold](https://github.com/adamreichold) [#1320](https://github.com/rust-ndarray/ndarray/pull/1320)
- small formatting fix in README.rst by [@podusowski](https://github.com/podusowski) [#1199](https://github.com/rust-ndarray/ndarray/pull/1199)
- Fix CI failures (mostly linting with clippy) by [@aganders3](https://github.com/aganders3) [#1171](https://github.com/rust-ndarray/ndarray/pull/1171)
- Remove doc(hidden) attr from items in trait impls by [@jturner314](https://github.com/jturner314) [#1165](https://github.com/rust-ndarray/ndarray/pull/1165)


Version 0.15.6 (2022-07-30)
===========================

New features
------------

- Add `get_ptr` and `get_mut_ptr` methods for getting an element's pointer from
  an index, by [@adamreichold].

  https://github.com/rust-ndarray/ndarray/pull/1151

Other changes
-------------

- Various fixes to resolve compiler and Clippy warnings/errors, by [@aganders3]
  and [@jturner314].

  https://github.com/rust-ndarray/ndarray/pull/1171

- Fix description of `stack!` in quick start docs, by [@jturner314]. Thanks to
  [@HyeokSuLee] for pointing out the issue.

  https://github.com/rust-ndarray/ndarray/pull/1156

- Add MSRV to `Cargo.toml`.

  https://github.com/rust-ndarray/ndarray/pull/1191


Version 0.15.5 (2022-07-30)
===========================

Enhancements
------------

- The `s!` macro now works in `no_std` environments, by [@makotokato].

  https://github.com/rust-ndarray/ndarray/pull/1154

Other changes
-------------

- Improve docs and fix typos, by [@steffahn] and [@Rikorose].

  https://github.com/rust-ndarray/ndarray/pull/1134 <br>
  https://github.com/rust-ndarray/ndarray/pull/1164


Version 0.15.4 (2021-11-23)
===========================

The Dr. Turner release 🚀

New features
------------

- Complex matrix multiplication now uses BLAS ``cgemm``/``zgemm`` when
  enabled (and matrix layout allows), by [@ethanhs].

  https://github.com/rust-ndarray/ndarray/pull/1106

- Use `matrixmultiply` as fallback for complex matrix multiplication
  when BLAS is not available or the matrix layout requires it by [@bluss]

  https://github.com/rust-ndarray/ndarray/pull/1118

- Add ``into/to_slice_memory_order`` methods for views, lifetime-preserving
  versions of existing similar methods by [@jturner314]

  https://github.com/rust-ndarray/ndarray/pull/1015

- ``kron`` function for Kronecker product by [@ethanhs].

  https://github.com/rust-ndarray/ndarray/pull/1105

- ``split_complex`` method for splitting complex arrays into separate
  real and imag view parts by [@jturner314] and [@ethanhs].

  https://github.com/rust-ndarray/ndarray/pull/1107

- New method ``try_into_owned_nocopy`` by [@jturner314]

  https://github.com/rust-ndarray/ndarray/pull/1022

- New producer and iterable ``axis_windows`` by [@VasanthakumarV]
  and [@jturner314].

  https://github.com/rust-ndarray/ndarray/pull/1022

- New method ``Zip::par_fold`` by [@adamreichold]

  https://github.com/rust-ndarray/ndarray/pull/1095

- New constructor ``from_diag_elem`` by [@jturner314]

  https://github.com/rust-ndarray/ndarray/pull/1076

- ``Parallel::with_min_len`` method for parallel iterators by [@adamreichold]

  https://github.com/rust-ndarray/ndarray/pull/1081

- Allocation-preserving map function ``.mapv_into_any()`` added by [@benkay86]

Enhancements
------------

- Improve performance of ``.sum_axis()`` for some cases by [@jturner314]

  https://github.com/rust-ndarray/ndarray/pull/1061

Bug fixes
---------

- Fix error in calling dgemv (matrix-vector multiplication) with BLAS and
  broadcasted arrays, by [@jturner314].

  https://github.com/rust-ndarray/ndarray/pull/1088

API changes
-----------

- Support approx 0.5 partially alongside the already existing approx 0.4 support.
  New feature flag is `approx-0_5`, by [@jturner314]

  https://github.com/rust-ndarray/ndarray/pull/1025

- Slice and reference-to-array conversions to CowArray added for by [@jturner314].

  https://github.com/rust-ndarray/ndarray/pull/1038

- Allow trailing comma in stack and concatenate macros by [@jturner314]

  https://github.com/rust-ndarray/ndarray/pull/1044

- ``Zip`` now has a ``must_use`` marker to help users by [@adamreichold]

  https://github.com/rust-ndarray/ndarray/pull/1082

Other changes
-------------

- Fixing the crates.io badge on github by [@atouchet]

  https://github.com/rust-ndarray/ndarray/pull/1104

- Use intra-doc links in docs by [@LeSeulArtichaut]

  https://github.com/rust-ndarray/ndarray/pull/1033

- Clippy fixes by [@adamreichold]

  https://github.com/rust-ndarray/ndarray/pull/1092 <br>
  https://github.com/rust-ndarray/ndarray/pull/1091

- Minor fixes in links and punctuation in docs by [@jimblandy]

  https://github.com/rust-ndarray/ndarray/pull/1056

- Minor fixes in docs by [@chohner]

  https://github.com/rust-ndarray/ndarray/pull/1119

- Update tests to quickcheck 1.0 by [@bluss]

  https://github.com/rust-ndarray/ndarray/pull/1114


Version 0.15.3 (2021-06-05)
===========================

New features
------------

- New methods `.last/_mut()` for arrays and array views by [@jturner314]

  https://github.com/rust-ndarray/ndarray/pull/1013

Bug fixes
---------

- Fix `as_slice_memory_order_mut()` so that it never changes strides (the
  memory layout) of the array when called.

  This was a bug that impacted `ArcArray` (and for example not `Array` or `ArrayView/Mut`),
  and multiple methods on `ArcArray` that use `as_slice_memory_order_mut` (for example `map_mut`).
  Fix by [@jturner314].

  https://github.com/rust-ndarray/ndarray/pull/1019

API changes
-----------

- Array1 now implements `From<Box<[T]>>` by [@jturner314]

  https://github.com/rust-ndarray/ndarray/pull/1016

- ArcArray now implements `From<Array<...>>` by [@jturner314]

  https://github.com/rust-ndarray/ndarray/pull/1021

- CowArray now implements RawDataSubst by [@jturner314]

  https://github.com/rust-ndarray/ndarray/pull/1020

Other changes
-------------

- Mention unsharing in `.as_mut_ptr` docs by [@jturner314]

  https://github.com/rust-ndarray/ndarray/pull/1017

- Clarify and fix minor errors in push/append method docs by [@bluss] f21c668a

- Fix several warnings in doc example code by [@bluss]

  https://github.com/rust-ndarray/ndarray/pull/1009


Version 0.15.2 (2021-05-17 🇳🇴)
================================

New features
------------

- New methods for growing/appending to owned `Array`s. These methods allow
  building an array efficiently chunk by chunk. By [@bluss].

  - `.push_row()`, `.push_column()`
  - `.push(axis, array)`, `.append(axis, array)`

  `stack`, `concatenate` and `.select()` now support all `Clone`-able elements
  as a result.

  https://github.com/rust-ndarray/ndarray/pull/932 <br>
  https://github.com/rust-ndarray/ndarray/pull/990

- New reshaping method `.to_shape(...)`, called with new shape and optional
  ordering parameter, this is the first improvement for reshaping in terms of
  added features and increased consistency, with more to come. By [@bluss].

  https://github.com/rust-ndarray/ndarray/pull/982

- `Array` now implements a by-value iterator, by [@bluss].

  https://github.com/rust-ndarray/ndarray/pull/986

- New methods `.move_into()` and `.move_into_uninit()` which allow assigning
  into an array by moving values from an array into another, by [@bluss].

  https://github.com/rust-ndarray/ndarray/pull/932 <br>
  https://github.com/rust-ndarray/ndarray/pull/997

- New method `.remove_index()` for owned arrays by [@bluss]

  https://github.com/rust-ndarray/ndarray/pull/967

- New constructor `build_uninit` which makes it easier to initialize
  uninitialized arrays in a way that's generic over all owned array kinds.
  By [@bluss].

  https://github.com/rust-ndarray/ndarray/pull/1001

Enhancements
------------

- Preserve the allocation of the input array in some more cases for arithmetic ops by [@SparrowLii]

  https://github.com/rust-ndarray/ndarray/pull/963

- Improve broadcasting performance for &array + &array arithmetic ops by [@SparrowLii]

  https://github.com/rust-ndarray/ndarray/pull/965

Bug fixes
---------

- Fix an error in construction of empty array with negative strides, by [@jturner314].

  https://github.com/rust-ndarray/ndarray/pull/998

- Fix minor performance bug with loop order selection in Zip by [@bluss]

  https://github.com/rust-ndarray/ndarray/pull/977

API changes
-----------

- Add dimension getters to `Shape` and `StrideShape` by [@stokhos]

  https://github.com/rust-ndarray/ndarray/pull/978

Other changes
-------------

- Rustdoc now uses the ndarray logo that [@jturner314] created previously

  https://github.com/rust-ndarray/ndarray/pull/981

- Minor doc changes by [@stokhos], [@cassiersg] and [@jturner314]

  https://github.com/rust-ndarray/ndarray/pull/968 <br>
  https://github.com/rust-ndarray/ndarray/pull/971 <br>
  https://github.com/rust-ndarray/ndarray/pull/974

- A little refactoring to reduce generics bloat in a few places by [@bluss].

  https://github.com/rust-ndarray/ndarray/pull/1004


Version 0.15.1 (2021-03-29)
===========================

Enhancements
------------

- Arrays and views now have additional PartialEq impls so that it's possible to
  compare arrays with references to arrays and vice versa by [@bluss]

  https://github.com/rust-ndarray/ndarray/pull/958

Bug fixes
---------

- Fix panic in creation of `.windows()` producer from negative stride array by
  [@bluss]

  https://github.com/rust-ndarray/ndarray/pull/957

Other changes
-------------

- Update BLAS documentation further by @bluss

  https://github.com/rust-ndarray/ndarray/pull/955 <br>
  https://github.com/rust-ndarray/ndarray/pull/959


Version 0.15.0 (2021-03-25)
===========================

New features
------------

- Support inserting new axes while slicing by [@jturner314]. This is an example:

  ```rust
  let view = arr.slice(s![.., -1, 2..;-1, NewAxis]);
  ```

  https://github.com/rust-ndarray/ndarray/pull/570

- Support two-sided broadcasting in arithmetic operations with arrays by [@SparrowLii]

  This now allows, for example, addition of a 3 x 1 with a 1 x 3 array; the
  operands are in this case broadcast to 3 x 3 which is the shape of the result.

  Note that this means that a new trait bound is required in some places when
  mixing dimensionality types of arrays in arithmetic operations.

  https://github.com/rust-ndarray/ndarray/pull/898

- Support for compiling ndarray as `no_std` (using core and alloc) by
  [@xd009642] and [@bluss]

  https://github.com/rust-ndarray/ndarray/pull/861 <br>
  https://github.com/rust-ndarray/ndarray/pull/889

- New methods `.cell_view()` and `ArrayViewMut::into_cell_view` that enable
  new ways of working with array elements as if they were in Cells - setting
  elements through shared views and broadcast views, by [@bluss].

  https://github.com/rust-ndarray/ndarray/pull/877

- New methods `slice_each_axis/_mut/_inplace` that make it easier to slice
  a dynamic number of axes in some situations, by [@jturner314]

  https://github.com/rust-ndarray/ndarray/pull/913

- New method `a.assign_to(b)` with the inverse argument order compared to the
  existing `b.assign(a)` and some extra features like assigning into
  uninitialized arrays, By [@bluss].

  https://github.com/rust-ndarray/ndarray/pull/947

- New methods `.std()` and `.var()` for standard deviation and variance by
  [@kdubovikov]

  https://github.com/rust-ndarray/ndarray/pull/790

Enhancements
------------

- Ndarray can now correctly determine that arrays can be contiguous, even if
  they have negative strides, by [@SparrowLii]

  https://github.com/rust-ndarray/ndarray/pull/885 <br>
  https://github.com/rust-ndarray/ndarray/pull/948

- Improvements to `map_inplace` by [@jturner314]

  https://github.com/rust-ndarray/ndarray/pull/911

- `.into_dimensionality` performance was improved for the `IxDyn` to `IxDyn`
  case by [@bluss]

  https://github.com/rust-ndarray/ndarray/pull/906

- Improved performance for scalar + &array and &array + scalar operations by
  [@jturner314]

  https://github.com/rust-ndarray/ndarray/pull/890

API changes
-----------

- New constructors `Array::from_iter` and `Array::from_vec` by [@bluss].
  No new functionality, just that these constructors are available without trait
  imports.

  https://github.com/rust-ndarray/ndarray/pull/921

- `NdProducer::raw_dim` is now a documented method by [@jturner314]

  https://github.com/rust-ndarray/ndarray/pull/918

- `AxisDescription` is now a struct with field names, not a tuple struct by
  [@jturner314]. Its accessor methods are now deprecated.

  https://github.com/rust-ndarray/ndarray/pull/915

- Methods for array comparison `abs_diff_eq` and `relative_eq` are now
  exposed as inherent methods too (no trait import needed), still under the approx
  feature flag by [@bluss]

  https://github.com/rust-ndarray/ndarray/pull/946

- Changes to the slicing-related types and macro by [@jturner314] and [@bluss]:

  - Remove the `Dimension::SliceArg` associated type, and add a new `SliceArg`
    trait for this purpose.
  - Change the return type of the `s![]` macro to an owned `SliceInfo` rather
    than a reference.
  - Replace the `SliceOrIndex` enum with `SliceInfoElem`, which has an
    additional `NewAxis` variant and does not have a `step_by` method.
  - Change the type parameters of `SliceInfo` in order to support the `NewAxis`
    functionality and remove some tricky `unsafe` code.
  - Mark the `SliceInfo::new` method as `unsafe`. The new implementations of
    `TryFrom` can be used as a safe alternative.
  - Remove the `AsRef<SliceInfo<[SliceOrIndex], D>> for SliceInfo<T, D>`
    implementation. Add the similar `From<&'a SliceInfo<T, Din, Dout>> for
    SliceInfo<&'a [SliceInfoElem], Din, Dout>` conversion as an alternative.
  - Change the *expr* `;` *step* case in the `s![]` macro to error at compile
    time if an unsupported type for *expr* is used, instead of panicking at
    runtime.

  https://github.com/rust-ndarray/ndarray/pull/570 <br>
  https://github.com/rust-ndarray/ndarray/pull/940 <br>
  https://github.com/rust-ndarray/ndarray/pull/943 <br>
  https://github.com/rust-ndarray/ndarray/pull/945 <br>

- Removed already deprecated methods by [@bluss]:

  - Remove deprecated `.all_close()` - use approx feature and methods like  `.abs_diff_eq` instead
  - Mark `.scalar_sum()` as deprecated - use `.sum()` instead
  - Remove deprecated `DataClone` - use `Data + RawDataClone` instead
  - Remove deprecated `ArrayView::into_slice` - use `to_slice()` instead.

  https://github.com/rust-ndarray/ndarray/pull/874

- Remove already deprecated methods: rows, cols (for row and column count; the
  new names are nrows and ncols) by [@bluss]

  https://github.com/rust-ndarray/ndarray/pull/872

- Renamed `Zip` methods by [@bluss] and [@SparrowLii]:

  - `apply` -> `for_each` 
  - `apply_collect` -> `map_collect`
  - `apply_collect_into` -> `map_collect_into`
  - (`par_` prefixed methods renamed accordingly)

  https://github.com/rust-ndarray/ndarray/pull/894 <br>
  https://github.com/rust-ndarray/ndarray/pull/904 <br>

- Deprecate `Array::uninitialized` and revamped its replacement by [@bluss]

  Please use new new `Array::uninit` which is based on `MaybeUninit` (renamed
  from `Array::maybe_uninit`, the old name is also deprecated).

  https://github.com/rust-ndarray/ndarray/pull/902 <br>
  https://github.com/rust-ndarray/ndarray/pull/876

- Renamed methods (old names are now deprecated) by [@bluss] and [@jturner314]

  - `genrows/_mut` -> `rows/_mut`
  - `gencolumns/_mut` -> `columns/_mut`
  - `stack_new_axis` -> `stack` (the new name already existed)
  - `visit` -> `for_each`

  https://github.com/rust-ndarray/ndarray/pull/872 <br>
  https://github.com/rust-ndarray/ndarray/pull/937 <br>
  https://github.com/rust-ndarray/ndarray/pull/907 <br>

- Updated `matrixmultiply` dependency to 0.3.0 by [@bluss]
  and adding new feature flag `matrixmultiply-threading` to enable its threading

  https://github.com/rust-ndarray/ndarray/pull/888 <br>
  https://github.com/rust-ndarray/ndarray/pull/938 <br>
 
- Updated `num-complex` dependency to 0.4.0 by [@bluss]

  https://github.com/rust-ndarray/ndarray/pull/952

Bug fixes
---------

- Fix `Zip::indexed` for the 0-dimensional case by [@jturner314]

  https://github.com/rust-ndarray/ndarray/pull/862

- Fix bug in layout computation that broke parallel collect to f-order
  array in some circumstances by [@bluss]

  https://github.com/rust-ndarray/ndarray/pull/900

- Fix an unwanted panic in shape overflow checking by [@bluss]

  https://github.com/rust-ndarray/ndarray/pull/855

- Mark the `SliceInfo::new` method as `unsafe` due to the requirement that
  `indices.as_ref()` always return the same value when called multiple times,
  by [@bluss] and [@jturner314]

  https://github.com/rust-ndarray/ndarray/pull/570

Other changes
-------------

- It was changed how we integrate with BLAS and `blas-src`. Users of BLAS need
  to read the README for the updated instructions. Ndarray itself no longer
  has public dependency on `blas-src`. Changes by [@bluss].

  https://github.com/rust-ndarray/ndarray/pull/891 <br>
  https://github.com/rust-ndarray/ndarray/pull/951

- Various improvements to tests and CI by [@jturner314]

  https://github.com/rust-ndarray/ndarray/pull/934 <br>
  https://github.com/rust-ndarray/ndarray/pull/924 <br>

- The `sort-axis.rs` example file's implementation of sort was bugfixed and now
  has tests, by [@dam5h] and [@bluss]

  https://github.com/rust-ndarray/ndarray/pull/916 <br>
  https://github.com/rust-ndarray/ndarray/pull/930

- We now link to the #rust-sci room on matrix in the readme by [@jturner314]

  https://github.com/rust-ndarray/ndarray/pull/619

- Internal cleanup with builder-like methods for creating arrays by [@bluss]

  https://github.com/rust-ndarray/ndarray/pull/908

- Implementation fix of `.swap(i, j)` by [@bluss]

  https://github.com/rust-ndarray/ndarray/pull/903

- Minimum supported Rust version (MSRV) is Rust 1.49.

  https://github.com/rust-ndarray/ndarray/pull/902

- Minor improvements to docs by [@insideoutclub]

  https://github.com/rust-ndarray/ndarray/pull/887


Version 0.14.0 (2020-11-28)
===========================

New features
------------

- `Zip::apply_collect` and `Zip::par_apply_collect` now support all
  elements (not just `Copy` elements) by [@bluss]
  https://github.com/rust-ndarray/ndarray/pull/814  
  https://github.com/rust-ndarray/ndarray/pull/817

- New function `stack` by [@andrei-papou]  
  https://github.com/rust-ndarray/ndarray/pull/844  
  https://github.com/rust-ndarray/ndarray/pull/850

Enhancements
------------

- Handle inhomogeneous shape inputs better in Zip, in practice: guess better whether
  to prefer c- or f-order for the inner loop by [@bluss]
  https://github.com/rust-ndarray/ndarray/pull/809

- Improve code sharing in some commonly used code by [@bluss]
  https://github.com/rust-ndarray/ndarray/pull/819

API changes
-----------

- The **old function** `stack` has been renamed to `concatenate`.
  A new function `stack` with numpy-like semantics have taken its place.
  Old usages of `stack` should change to use `concatenate`.
  
  `concatenate` produces an array with the same number of axes as the inputs.  
  `stack` produces an array that has one more axis than the inputs.

  This change was unfortunately done without a deprecation period, due to the long period between releases.

  https://github.com/rust-ndarray/ndarray/pull/844  
  https://github.com/rust-ndarray/ndarray/pull/850

- Enum ErrorKind is now properly non-exhaustive and has lost its old placeholder invalid variant. By [@Zuse64]
  https://github.com/rust-ndarray/ndarray/pull/848

- Remove deprecated items:

  - RcArray (deprecated alias for ArcArray)
  - Removed `subview_inplace` use `collapse_axis`
  - Removed `subview_mut` use `index_axis_mut`
  - Removed `into_subview` use `index_axis_move`
  - Removed `subview` use `index_axis`
  - Removed `slice_inplace` use `slice_collapse`

- Undeprecated `remove_axis` because its replacement is hard to find out on your own.

- Update public external dependencies to new versions by [@Eijebong] and [@bluss]

  - num-complex 0.3
  - approx 0.4 (optional)
  - blas-src 0.6.1 and openblas-src 0.9.0 (optional)

  https://github.com/rust-ndarray/ndarray/pull/810  
  https://github.com/rust-ndarray/ndarray/pull/851  


Other changes
-------------

- Minor doc fixes by [@acj]
  https://github.com/rust-ndarray/ndarray/pull/834

- Minor doc fixes by [@xd009642]
  https://github.com/rust-ndarray/ndarray/pull/847

- The minimum required rust version is Rust 1.42.

- Release management by [@bluss]

Version 0.13.1 (2020-04-21)
===========================

New features
------------

- New *amazing* slicing methods `multi_slice_*` by [@jturner314]
  https://github.com/rust-ndarray/ndarray/pull/717
- New method `.cast()` for raw views by [@bluss]
  https://github.com/rust-ndarray/ndarray/pull/734
- New aliases `ArcArray1`, `ArcArray2` by [@d-dorazio]
  https://github.com/rust-ndarray/ndarray/pull/741
- New array constructor `from_shape_simple_fn` by [@bluss]
  https://github.com/rust-ndarray/ndarray/pull/728
- `Dimension::Larger` now requires `RemoveAxis` by [@TheLortex]
  https://github.com/rust-ndarray/ndarray/pull/792
- New methods for collecting Zip into an array by [@bluss]
  https://github.com/rust-ndarray/ndarray/pull/797
- New `Array::maybe_uninit` and `.assume_init()` by [@bluss]
  https://github.com/rust-ndarray/ndarray/pull/803

Enhancements
------------

- Remove itertools as dependency by [@bluss]
  https://github.com/rust-ndarray/ndarray/pull/730
- Improve `zip_mut_with` (and thus arithmetic ops) for f-order arrays by [@nilgoyette]
  https://github.com/rust-ndarray/ndarray/pull/754
- Implement `fold` for `IndicesIter` by [@jturner314]
  https://github.com/rust-ndarray/ndarray/pull/733
- New Quick Start readme by [@lifuyang]
  https://github.com/rust-ndarray/ndarray/pull/785

API changes
-----------

- Remove alignment restriction on raw views by [@jturner314]
  https://github.com/rust-ndarray/ndarray/pull/738

Other changes
-------------

- Fix documentation in ndarray for numpy users by [@jturner314]
- Improve blas version documentation by [@jturner314]
- Doc improvements by [@mockersf] https://github.com/rust-ndarray/ndarray/pull/751
- Doc and lint related improvements by [@viniciusd] https://github.com/rust-ndarray/ndarray/pull/750
- Minor fixes related to best practices for unsafe code by [@bluss]
  https://github.com/rust-ndarray/ndarray/pull/799
  https://github.com/rust-ndarray/ndarray/pull/802
- Release management by [@bluss]


Version 0.13.0 (2019-09-23)
===========================

New features
------------

 - `ndarray-parallel` is merged into `ndarray`. Use the `rayon` feature-flag to get access to parallel iterators and
   other parallelized methods.
   ([#563](https://github.com/rust-ndarray/ndarray/pull/563/files) by [@bluss])
 - Add `logspace` and `geomspace` constructors
   ([#617](https://github.com/rust-ndarray/ndarray/pull/617) by [@JP-Ellis])
 - Implement approx traits for `ArrayBase`. They can be enabled using the `approx` feature-flag.
   ([#581](https://github.com/rust-ndarray/ndarray/pull/581) by [@jturner314])
 - Add `mean` method
   ([#580](https://github.com/rust-ndarray/ndarray/pull/580) by [@LukeMathWalker])
 - Add `Zip::all` to check if all elements satisfy a predicate
   ([#615](https://github.com/rust-ndarray/ndarray/pull/615) by [@mneumann])
 - Add `RawArrayView` and `RawArrayViewMut` types and `RawData`, `RawDataMut`, and `RawDataClone` traits
   ([#496](https://github.com/rust-ndarray/ndarray/pull/496) by [@jturner314])
 - Add `CowArray`, `C`lone `o`n `write` array
   ([#632](https://github.com/rust-ndarray/ndarray/pull/632) by [@jturner314] and [@andrei-papou])
 - Add `as_standard_layout` to `ArrayBase`: it takes an array by reference and returns a `CoWArray` in standard layout
   ([#616](https://github.com/rust-ndarray/ndarray/pull/616) by [@jturner314] and [@andrei-papou])
 - Add `Array2::from_diag` method to create 2D arrays from a diagonal
   ([#673](https://github.com/rust-ndarray/ndarray/pull/673) by [@rth])
 - Add `fold` method to `Zip`
   ([#684](https://github.com/rust-ndarray/ndarray/pull/684) by [@jturner314])
 - Add `split_at` method to `AxisChunksIter/Mut`
   ([#691](https://github.com/rust-ndarray/ndarray/pull/691) by [@jturner314])
 - Implement parallel iteration for `AxisChunksIter/Mut`
   ([#639](https://github.com/rust-ndarray/ndarray/pull/639) by [@nitsky])
 - Add `into_scalar` method to `ArrayView0` and `ArrayViewMut0`
   ([#700](https://github.com/rust-ndarray/ndarray/pull/700) by [@LukeMathWalker])
 - Add `accumulate_axis_inplace` method to `ArrayBase`
   ([#611](https://github.com/rust-ndarray/ndarray/pull/611) by [@jturner314] and [@bluss])
 - Add the `array!`, `azip!`, and `s!` macros to `ndarray::prelude`
   ([#517](https://github.com/rust-ndarray/ndarray/pull/517) by [@jturner314])

Enhancements
------------
 - Improve performance for matrix multiplications when using the pure-Rust backend thanks to `matrix-multiply:v0.2` 
   (leverage SIMD instructions on x86-64 with runtime feature detection)
   ([#556](https://github.com/rust-ndarray/ndarray/pull/556) by [@bluss])
 - Improve performance of `fold` for iterators
   ([#574](https://github.com/rust-ndarray/ndarray/pull/574) by [@jturner314])
 - Improve performance of `nth_back` for iterators
   ([#686](https://github.com/rust-ndarray/ndarray/pull/686) by [@jturner314])
 - Improve performance of iterators for 1-d arrays
   ([#614](https://github.com/rust-ndarray/ndarray/pull/614) by [@andrei-papou])
 - Improve formatting for large arrays
   ([#606](https://github.com/rust-ndarray/ndarray/pull/606) by [@andrei-papou] and [@LukeMathWalker],
   [#633](https://github.com/rust-ndarray/ndarray/pull/633) and [#707](https://github.com/rust-ndarray/ndarray/pull/707) by [@jturner314],
   and [#713](https://github.com/rust-ndarray/ndarray/pull/713) by [@bluss])
 - Arithmetic operations between arrays with different element types are now allowed when there is a scalar equivalent
   ([#588](https://github.com/rust-ndarray/ndarray/pull/588) by [@jturner314])
 - `.map_axis/_mut` won't panic on 0-length `axis`
   ([#579](https://github.com/rust-ndarray/ndarray/pull/612) by [@andrei-papou])
 - Various documentation improvements (by [@jturner314], [@JP-Ellis], [@LukeMathWalker], [@bluss])

API changes
-----------
 - The `into_slice` method on ArrayView is deprecated and renamed to `to_slice`
   ([#646](https://github.com/rust-ndarray/ndarray/pull/646) by [@max-sixty])
 - `RcArray` is deprecated in favour of `ArcArray` 
   ([#560](https://github.com/rust-ndarray/ndarray/pull/560) by [@bluss])
 - `into_slice` is renamed to `to_slice`. `into_slice` is now deprecated
   ([#646](https://github.com/rust-ndarray/ndarray/pull/646) by [@max-sixty])
 - `from_vec` is deprecated in favour of using the `From` to convert a `Vec` into an `Array`
   ([#648](https://github.com/rust-ndarray/ndarray/pull/648) by [@max-sixty])
 - `mean_axis` returns `Option<A>` instead of `A`, to avoid panicking when invoked on a 0-length axis 
   ([#580](https://github.com/rust-ndarray/ndarray/pull/580) by [@LukeMathWalker])
 - Remove `rustc-serialize` feature-flag. `serde` is the recommended feature-flag for serialization
   ([#557](https://github.com/rust-ndarray/ndarray/pull/557) by [@bluss])
 - `rows`/`cols` are renamed to `nrows`/`ncols`. `rows`/`cols` are now deprecated
   ([#701](https://github.com/rust-ndarray/ndarray/pull/701) by [@bluss])
 - The usage of the `azip!` macro has changed to be more similar to `for` loops
   ([#626](https://github.com/rust-ndarray/ndarray/pull/626) by [@jturner314])
 - For `var_axis` and `std_axis`, the constraints on `ddof` and the trait bounds on `A` have been made more strict
   ([#515](https://github.com/rust-ndarray/ndarray/pull/515) by [@jturner314])
 - For `mean_axis`, the constraints on `A` have changed
   ([#518](https://github.com/rust-ndarray/ndarray/pull/518) by [@jturner314])
 - `DataClone` is deprecated in favor of using `Data + RawDataClone`
   ([#496](https://github.com/rust-ndarray/ndarray/pull/496) by [@jturner314])
 - The `Dimension::Pattern` associated type now has more trait bounds
   ([#634](https://github.com/rust-ndarray/ndarray/pull/634) by [@termoshtt])
 - `Axis::index()` now takes `self` instead of `&self`
   ([#642](https://github.com/rust-ndarray/ndarray/pull/642) by [@max-sixty])
 - The bounds on the implementation of `Hash` for `Dim` have changed
   ([#642](https://github.com/rust-ndarray/ndarray/pull/642) by [@max-sixty])

Bug fixes
---------
 - Prevent overflow when computing strides in `do_slice`
   ([#575](https://github.com/rust-ndarray/ndarray/pull/575) by [@jturner314])
 - Fix issue with BLAS matrix-vector multiplication for array with only 1 non-trivial dimension
   ([#585](https://github.com/rust-ndarray/ndarray/pull/585) by [@sebasv])
 - Fix offset computation to avoid UB/panic when slicing in some edge cases
   ([#636](https://github.com/rust-ndarray/ndarray/pull/636) by [@jturner314])
 - Fix issues with axis iterators
   ([#669](https://github.com/rust-ndarray/ndarray/pull/669) by [@jturner314])
 - Fix handling of empty input to `s!` macro
   ([#714](https://github.com/rust-ndarray/ndarray/pull/714) by [@bluss] and [#715](https://github.com/rust-ndarray/ndarray/pull/715) by [@jturner314])

Other changes
-------------
 - Various improvements to `ndarray`'s CI pipeline (`clippy`, `cargo fmt`, etc. by [@max-sixty] and [@termoshtt])
 - Bump minimum required Rust version to 1.37.


Version 0.12.1 (2018-11-21)
===========================

  - Add `std_axis` method for computing standard deviation by @LukeMathWalker.
  - Add `product` method for computing product of elements in an array by @sebasv.
  - Add `first` and `first_mut` methods for getting the first element of an array.
  - Add `into_scalar` method for converting an `Array0` into its element.
  - Add `insert_axis_inplace` and `index_axis_inplace` methods for inserting and
    removing axes in dynamic-dimensional (`IxDyn`) arrays without taking ownership.
  - Add `stride_of` method for getting the stride of an axis.
  - Add public `ndim` and `zeros` methods to `Dimension` trait.
  - Rename `scalar_sum` to `sum`, `subview` to `index_axis`,
    `subview_mut` to `index_axis_mut`, `subview_inplace` to
    `collapse_axis`, `into_subview` to `index_axis_move`, and
    `slice_inplace` to `slice_collapse` (deprecating the old names,
    except for `scalar_sum` which will be in 0.13).
  - Deprecate `remove_axis` and fix soundness hole when removing a zero-length axis.
  - Implement `Clone` for `LanesIter`.
  - Implement `Debug`, `Copy`, and `Clone` for `FoldWhile`.
  - Relax constraints on `sum_axis`, `mean_axis`, and `into_owned`.
  - Add number of dimensions (and whether it's const or dynamic) to array `Debug` format.
  - Allow merging axes with `merge_axes` when either axis length is ≤ 1.
  - Clarify and check more precise safety requirements for constructing arrays.
    This fixes undefined behavior in some edge cases.
    (See [#543](https://github.com/rust-ndarray/ndarray/pull/543).)
  - Fix `is_standard_layout` in some edge cases.
    (See [#543](https://github.com/rust-ndarray/ndarray/pull/543).)
  - Fix chunk sizes in `axis_chunks_iter` and `axis_chunks_iter_mut` when
    the stride is zero or the array element type is zero-sized by @bluss.
  - Improve documentation by @jturner314, @bluss, and @paulkernfeld.
  - Improve element iterators with implementations of `Iterator::rfold`.
  - Miscellaneous internal implementation improvements by @jturner314 and @bluss.


Version 0.12.0 (2018-09-01)
===========================

  - Add `var_axis` method for computing variance by @LukeMathWalker.
  - Add `map_mut` and `map_axis_mut` methods (mutable variants of `map` and `map_axis`) by @LukeMathWalker.
  - Add support for 128-bit integer scalars (`i128` and `u128`).
  - Add support for slicing with inclusive ranges (`start..=end` and `..=end`).
  - Relax constraint on closure from `Fn` to `FnMut` for `mapv`, `mapv_into`, `map_inplace` and `mapv_inplace`.
  - Implement `TrustedIterator` for `IterMut`.
  - Bump `num-traits` and `num-complex` to version `0.2`.
  - Bump `blas-src` to version `0.2`.
  - Bump minimum required Rust version to 1.27.
  - Additional contributors to this release: @ExpHP, @jturner314, @alexbool, @messense, @danmack, @nbro

Version 0.11.2 (2018-03-21)
===========================

  - New documentation; @jturner314 has written a large “ndarray for NumPy users”
    document, which we include in rustdoc.
    [Read it here](https://docs.rs/ndarray/0.11/ndarray/doc/ndarray_for_numpy_users/)
    a useful quick guide for any user, and in particular if you are familiar
    with numpy.
  - Add `ArcArray`. `RcArray` has become `ArcArray`; it is now using thread
    safe reference counting just like `Arc`; this means that shared ownership
    arrays are now `Send/Sync` if the corresponding element type is `Send
    + Sync`.
  - Add array method `.permute_axes()` by @jturner314
  - Add array constructor `Array::ones` by @ehsanmok
  - Add the method `.reborrow()` to `ArrayView/Mut`, which can be used
    to shorten the lifetime of an array view; in a reference-like type this
    normally happens implicitly but for technical reasons the views have
    an invariant lifetime parameter.
  - Fix an issue with type inference, the dimensionality of an array
    should not infer correctly in more cases when using slicing. By @jturner314.


Version 0.11.1 (2018-01-21)
===========================

  - Dimension types (`Ix1, Ix2, .., IxDyn`) now implement `Hash` by
    @jturner314
  - Blas integration can now use *gemv* for matrix-vector multiplication also
    when the matrix is f-order by @maciejkula
  - Encapsulated `unsafe` code blocks in the `s![]` macro are now exempted
    from the `unsafe_code` lint by @jturner314

Version 0.11.0 (2017-12-29)
===========================

[Release announcement](https://jim.turner.link/pages/ndarray-0.11/)

  - Allow combined slicing and subviews in a single operation by @jturner314 and
    @bluss

    * Add support for individual indices (to indicate subviews) to the `s![]`
      macro, and change the return type to
      `&SliceInfo<[SliceOrIndex; n], Do>`.
    * Change the argument type of the slicing methods to correspond to the new
      `s![]` macro.
    * Replace the `Si` type with `SliceOrIndex`.
    * Add a new `Slice` type that is similar to the old `Si` type.

  - Add support for more index types (e.g. `usize`) to the `s![]` macro by
    @jturner314
  - Rename `.islice()` to `.slice_inplace()` by @jturner314
  - Rename `.isubview()` to `.subview_inplace()` by @jturner314
  - Add `.slice_move()`, `.slice_axis()`, `.slice_axis_mut()`, and
    `.slice_axis_inplace()` methods by @jturner314
  - Add `Dimension::NDIM` associated constant by @jturner314
  - Change trait bounds for arithmetic ops between an array (by value) and
    a reference to an array or array view (“array1 (op) &array2”); before,
    an `ArrayViewMut` was supported on the left hand side, now, the left
    hand side must not be a view.
    ([#380](https://github.com/rust-ndarray/ndarray/pull/380)) by @jturner314
  - Remove deprecated methods (`.whole_chunks()`, `.whole_chunks_mut()`,
    `.sum()`, and `.mean()`; replaced by `.exact_chunks()`,
    `.exact_chunks_mut()`, `.sum_axis()`, and `.mean_axis()`,
    respectively) by @bluss
  - Updated to the latest blas (optional) dependencies. See instructions in the
    README.
  - Minimum required Rust version is 1.22.


Earlier releases
================

- 0.10.13

  - Add an extension trait for longer-life indexing methods for array views
    (`IndexLonger`) by @termoshtt and @bluss
  - The `a.dot(b)` method now supports a vector times matrix multiplication
    by @jturner314
  - More general `.into_owned()` method by @jturner314

- 0.10.12

  - Implement serde serialization for `IxDyn`, so that arrays and array views
    using it are serializable as well.

- 0.10.11

  - Add method `.uswap(a, b)` for unchecked swap by @jturner314
  - Bump private dependencies (itertools 0.7)

- 0.10.10

  - Fix crash with zero size arrays in the fallback matrix multiplication code
    (#365) by @jturner314

- 0.10.9

  - Fix crash in `Array::from_shape_fn` when creating an f-order array
    with zero elements (#361) by @jturner314

- 0.10.8

  - Add method `.insert_axis()` to arrays and array views by @jturner314

- 0.10.7

  - Add method `.is_empty()` to arrays and array views by @iamed2
  - Support optional trailing commas in the `array![]` macro by Alex Burka
  - Added an example of permuting/sorting along an axis to the sources

- 0.10.6

  - Tweak the implementation for (bounds checked) indexing of arrays
    ([] operator). The new code will have the optimizer elide the bounds checks
    in more situations.

- 0.10.5

  - Add method `.into_dimensionality::<D>()` for dimensionality conversion
    (From `IxDyn` to fixed size and back).
  - New names `.sum_axis` and `.mean_axis` for sum and mean functions.
    Old names deprecated to make room for scalar-returning methods, making
    a proper convention.
  - Fix deserialization using ron (#345) by @Libbum

- 0.10.4

  - Fix unused mut warnings in `azip!()` macro
  - Fix bug #340 by @lloydmeta; uses blas gemm for more memory layouts
    of column matrices. Only relevant if using blas.

- 0.10.3

  - Fix docs.rs doc build

- 0.10.2

  - Support trailing commas in the `s![]` macro
  - Some documentation improvements for the introduction, for `azip!()` and
    other places.
  - Added two more examples in the source

- 0.10.1

  - Add method `.into_dyn()` to convert to a dynamic dimensionality array
    or array view. By @bobogei81123
  - Edit docs for the fact that type alias pages now show methods.
    See the doc pages for `Array` and `ArrayView` and the other aliases.
  - Edit docs for `Zip`

- 0.10.0

  - Upgrade to Serde 1.0. Crate feature name is `serde-1`.
  - Require Rust 1.18. The `pub(crate)` feature is that important.


- 0.9.1

  - Fix `Array::from_shape_fn` to give correct indices for f-order shapes
  - Fix `Array::from_shape_fn` to panic correctly on shape size overflow

- 0.9.0 [Release Announcement](https://bluss.github.io//rust/2017/04/09/ndarray-0.9/)

  - Add `Zip::indexed`
  - New methods `genrows/_mut, gencolumns/_mut, lanes/_mut` that
    return iterable producers (producer means `Zip` compatible).
  - New method `.windows()` by @Robbepop, returns an iterable producer
  - New function `general_mat_vec_mul` (with fast default and blas acceleration)
  - `Zip::apply` and `fold_while` now take `self` as the first argument
  - `indices/_of` now return iterable producers (not iterator)
  - No allocation for short `IxDyn`.
  - Remove `Ix, Ixs` from the prelude
  - Remove deprecated `Axis::axis` method (use `.index()`)
  - Rename `.whole_chunks` to `.exact_chunks`.
  - Remove `.inner_iter` in favour of the new `.genrows()` method.
  - Iterators and similar structs are now scoped under `ndarray::iter`
  - `IntoNdProducer` now has the `Item` associated type
  - Owned array storage types are now encapsulated in newtypes
  - `FoldWhile` got the method `is_done`.
  - Arrays now implement formatting trait `Binary` if elements do
  - Internal changes. `NdProducer` generalized. `Dimension` gets
    the `Smaller` type parameter. Internal traits have the private marker now.
  - `#` (alternate) in formatting does nothing now.
  - Require Rust 1.15

- 0.8.4

  - Use `Zip` in `.all_close()` (performance improvement)
  - Use `#[inline]` on a function used for higher dimensional checked
    indexing (performance improvement for arrays of ndim >= 3)
  - `.subview()` has a more elaborate panic message

- 0.8.3

  - Fix a bug in `Zip` / `NdProducer` if an array of at least 3 dimensions
    was contig but not c- nor f-contig.
  - `WholeChunksIter/Mut` now impl `Send/Sync` as appropriate
  - Misc cleanup and using dimension-reducing versions of inner_iter
    internally. Remove a special case in `zip_mut_with` that only made it
    slower (1D not-contig arrays).

- 0.8.2

  - Add more documentation and an example for dynamic dimensions: see
    [`IxDyn`](https://docs.rs/ndarray/0.8.2/ndarray/type.IxDyn.html).
    `IxDyn` will have a representation change next incompatible
    version. Use it as a type alias for best forward compatibility.
  - Add iterable and producer `.whole_chunks_mut(size)`.
  - Fix a bug in `whole_chunks`: it didn't check the dimensionality of the
    requested chunk size properly (an `IxDyn`-only bug).
  - Improve performance of `zip_mut_with` (and thus all binary operators) for
    block slices of row major arrays.
  - `AxisChunksIter` creation sped up and it implements `Clone`.
  - Dimension mismatch in `Zip` has a better panic message.


- 0.8.1

  - Add `Zip` and macro `azip!()` which implement lock step function
    application across elements from one up to six arrays (or in general
    producers)

    + Apart from array views, axis iterators and the whole chunks iterable are
      also producers

  - Add constructor `Array::uninitialized`
  - Add iterable and producer `.whole_chunks(size)`
  - Implement a prettier `Debug` for `Si`.
  - Fix `Array::default` so that it panics as documented if the size of the
    array would wrap around integer type limits.
  - Output more verbose panics for errors when slicing arrays (only in debug
    mode).

- 0.8.0

  - Update serde dependency to 0.9
  - Remove deprecated type alias `OwnedArray` (use `Array`)
  - Remove deprecated `.assign_scalar()` (use `fill`)

- 0.7.3

  - Add macro `array![]` for creating one-, two-, or three-dimensional arrays
    (with ownership semantics like `vec![]`)
  - `Array` now implements `Clone::clone_from()` specifically, so that its
    allocation is (possibly) reused.
  - Add `.to_vec()` for one-dimensional arrays
  - Add `RcArray::into_owned(self) -> Array`.
  - Add crate categories

- 0.7.2

  - Add array methods `.remove_axis()`, `.merge_axes()` and `.invert_axis()`
  - Rename `Axis`’ accessor `axis` to `index`, old name is deprecated.

- 0.7.1

  - Fix two bugs in `Array::clone()`; it did not support zero-size elements
    like `()`, and for some negatively strided arrays it did not update the
    first element offset correctly.
  - Add `.axes()` which is an iterator over the axes of an array, yielding
    its index, length and stride.
  - Add method `.max_stride_axis()`.

- 0.6.10

  - Fix two bugs in `Array::clone()`; it did not support zero-size elements
    like `()`, and for some negatively strided arrays it did not update the
    first element offset correctly.

- 0.7.0

  - Big overhaul of dimensions: Add type `Dim` with aliases
    `Ix1, Ix2, Ix3, ...` etc for specific dimensionalities.
    Instead of `Ix` for dimension use `Ix1`, instead of `(Ix, Ix)` use
    `Ix2`, and so on.
  - The dimension type `Dim` supports indexing and arithmetic. See
    `Dimension` trait for new methods and inherited traits.
  - Constructors and methods that take tuples for array sizes, like `Array::zeros,`
    `Array::from_shape_vec`, `.into_shape()` and so on will continue to work
    with tuples.
  - The array method `.raw_dim()` returns the shape description
    `D` as it is. `.dim()` continues to return the dimension as a tuple.
  - Renamed iterators for consistency (each iterator is named for the
    method that creates it, for example `.iter()` returns `Iter`).
  - The index iterator is now created with free functions `indices` or
    `indices_of`.
  - Expanded the `ndarray::prelude` module with the dimensionality-specific
    type aliases, and some other items
  - `LinalgScalar` and related features no longer need to use `Any` for
    static type dispatch.
  - Serialization with `serde` now supports binary encoders like bincode
    and others.
  - `.assign_scalar()` was deprecated and replaced by `.fill()`, which
    takes an element by value.
  - Require Rust 1.13

- 0.6.9

  - Implement `ExactSizeIterator` for the indexed iterators

- 0.6.8

  - Fix a bug in a partially consumed elements iterator's `.fold()`.
    (**Note** that users are recommended to not use the elements iterator,
    but the higher level functions which are the maps, folds and other methods
    of the array types themselves.)

- 0.6.7

  - Improve performance of a lot of basic operations for arrays where
    the innermost dimension is not contiguous (`.fold(), .map(),
    .to_owned()`, arithmetic operations with scalars).
  - Require Rust 1.11

- 0.6.6

  - Add dimensionality specific type aliases: `Array0, Array1, Array2, ...`
    and so on (there are many), also `Ix0, Ix1, Ix2, ...`.
  - Add constructor `Array::from_shape_fn(D, |D| -> A)`.
  - Improve performance of `Array::default`, and `.fold()` for noncontiguous
    array iterators.

- 0.6.5

  - Add method `.into_raw_vec()` to turn an `Array` into the its
    underlying element storage vector, in whatever element order it is using.

- 0.6.4

  - Add method `.map_axis()` which is used to flatten an array along
    one axis by mapping it to a scalar.

- 0.6.3

  - Work around compilation issues in nightly (issue #217)
  - Add `Default` implementations for owned arrays

- 0.6.2

  - Add serialization support for serde 0.8, under the crate feature name `serde`

- 0.6.1

  - Add `unsafe` array view constructors `ArrayView::from_shape_ptr`
    for read-only and read-write array views. These make it easier to
    create views from raw pointers.

- 0.6.0

  - Rename `OwnedArray` to `Array`. The old name is deprecated.
  - Remove deprecated constructor methods. Use zeros, from_elem, from_shape_vec
    or from_shape_vec_unchecked instead.
  - Remove deprecated in place arithmetic methods like iadd et.c. Use += et.c.
    instead.
  - Remove deprecated method mat_mul, use dot instead.
  - Require Rust 1.9

- 0.5.2

  - Use num-traits, num-complex instead of num.

- 0.5.1

  - Fix theoretical well-formedness issue with Data trait

- 0.5.0

  - Require Rust 1.8 and enable +=, -=, and the other assign operators.
    All `iadd, iadd_scalar` and similar methods are now deprecated.
  - ndarray now has a prelude: `use ndarray::prelude::*;`.
  - Constructors from_elem, zeros, from_shape_vec now all support passing a custom
    memory layout. A lot of specific constructors were deprecated.
  - Add method `.select(Axis, &[Ix]) -> OwnedArray`, to create an array
    from a non-contiguous pick of subviews along an axis.
  - Rename `.mat_mul()` to just `.dot()` and add a function `general_mat_mul`
    for matrix multiplication with scaling into an existing array.
  - **Change .fold() to use arbitrary order.**
  - See below for more details

- 0.5.0-alpha.2

  - Fix a namespace bug in the stack![] macro.
  - Add method .select() that can pick an arbitrary set of rows (for example)
    into a new array.

- 0.4.9

  - Fix a namespace bug in the stack![] macro.
  - Add deprecation messages to .iadd() and similar methods (use += instead).

- 0.5.0-alpha.1

  - Add .swap(i, j) for swapping two elements
  - Add a prelude module `use ndarray::prelude::*;`
  - Add ndarray::linalg::general_mat_mul which computes *C ← α A B + β C*,
    i.e matrix multiplication into an existing array, with optional scaling.
  - Add .fold_axis(Axis, folder)
  - Implement .into_shape() for f-order arrays

- 0.5.0-alpha.0

  - Requires Rust 1.8. Compound assignment operators are now enabled by default.
  - Rename `.mat_mul()` to `.dot()`. The same method name now handles
    dot product and matrix multiplication.
  - Remove deprecated items: raw_data, raw_data_mut, allclose, zeros, Array.
    Docs for 0.4. lists the replacements.
  - Remove deprecated crate features: rblas, assign_ops
  - A few consuming arithmetic ops with ArrayViewMut were removed (this
    was missed in the last version).
  - **Change .fold() to use arbitrary order.** Its specification and
    implementation has changed, to pick the most appropriate element traversal
    order depending on memory layout.

- 0.4.8

  - Fix an error in `.dot()` when using BLAS and arrays with negative stride.

- 0.4.7

  - Add dependency matrixmultiply to handle matrix multiplication
    for floating point elements. It supports matrices of general stride
    and is a great improvement for performance. See PR #175.

- 0.4.6

  - Fix bug with crate feature blas; it would not compute matrix
    multiplication correctly for arrays with negative or zero stride.
  - Update blas-sys version (optional dependency).

- 0.4.5

  - Add `.all_close()` which replaces the now deprecated `.allclose()`.
    The new method has a stricter protocol: it panics if the array
    shapes are not compatible. We don't want errors to pass silently.
  - Add a new illustration to the doc for `.axis_iter()`.
  - Rename `OuterIter, OuterIterMut` to `AxisIter, AxisIterMut`.
    The old name is now deprecated.

- 0.4.4

  - Add mapping methods `.mapv(), .mapv_into(), .map_inplace(),`
    `.mapv_inplace(), .visit()`. The `mapv` versions
    have the transformation function receive the element by value (hence *v*).
  - Add method `.scaled_add()` (a.k.a axpy) and constructor `from_vec_dim_f`.
  - Add 2d array methods `.rows(), .cols()`.
  - Deprecate method `.fold()` because it dictates a specific visit order.

- 0.4.3

  - Add array method `.t()` as a shorthand to create a transposed view.
  - Fix `mat_mul` so that it accepts arguments of different array kind
  - Fix a bug in `mat_mul` when using BLAS and multiplying with a column
    matrix (#154)

- 0.4.2

  - Add new BLAS integration used by matrix multiplication
    (selected with crate feature `blas`). Uses pluggable backend.
  - Deprecate module `ndarray::blas` and crate feature `rblas`. This module
    was moved to the crate `ndarray-rblas`.
  - Add array methods `as_slice_memory_order, as_slice_memory_order_mut, as_ptr,
    as_mut_ptr`.
  - Deprecate `raw_data, raw_data_mut`.
  - Add `Send + Sync` to `NdFloat`.
  - Arrays now show shape & stride in their debug formatter.
  - Fix a bug where `from_vec_dim_stride` did not accept arrays with unitary axes.
  - Performance improvements for contiguous arrays in non-c order when using
    methods `to_owned, map, scalar_sum, assign_scalar`,
    and arithmetic operations between array and scalar.
  - Some methods now return arrays in the same memory order of the input
    if the input is contiguous: `to_owned, map, mat_mul` (matrix multiplication
    only if both inputs are the same memory order), and arithmetic operations
    that allocate a new result.
  - Slight performance improvements in `dot, mat_mul` due to more efficient
    glue code for calling BLAS.
  - Performance improvements in `.assign_scalar`.

- 0.4.1

  - Mark iterators `Send + Sync` when possible.

- **0.4.0** [Release Announcement](http://bluss.github.io/rust/2016/03/06/ndarray-0.4/)

  - New array splitting via `.split_at(Axis, Ix)` and `.axis_chunks_iter()`
  - Added traits `NdFloat`, `AsArray` and `From for ArrayView` which
    improve generic programming.
  - Array constructors panic when attempting to create an array whose element
    count overflows `usize`. (Would be a debug assertion for overflow before.)
  - Performance improvements for `.map()`.
  - Added `stack` and macro `stack![axis, arrays..]` to concatenate arrays.
  - Added constructor `OwnedArray::range(start, end, step)`.
  - The type alias `Array` was renamed to `RcArray` (and the old name deprecated).
  - Binary operators are not defined when consuming a mutable array view as
    the left hand side argument anymore.
  - Remove methods and items deprecated since 0.3 or earlier; deprecated methods
    have notes about replacements in 0.3 docs.
  - See below for full changelog through alphas.

- 0.4.0-alpha.8

  - In debug mode, indexing an array out of bounds now has a detailed
    message about index and shape. (In release mode it does not.)
  - Enable assign_ops feature automatically when it is supported (Rust 1.8 beta
    or later).
  - Add trait `NdFloat` which makes it easy to be generic over `f32, f64`.
  - Add `From` implementations that convert slices or references to arrays
    into array views. This replaces `from_slice` from a previous alpha.
  - Add `AsArray` trait, which is simply based on those `From` implementations.
  - Improve `.map()` so that it can autovectorize.
  - Use `Axis` argument in `RemoveAxis` too.
  - Require `DataOwned` in the raw data methods.
  - Merged error types into a single `ShapeError`, which uses no allocated data.

- 0.4.0-alpha.7

  - Fix too strict lifetime bound in arithmetic operations like `&a @ &b`.
  - Rename trait Scalar to ScalarOperand (and improve its docs).
  - Implement <<= and >>= for arrays.

- 0.4.0-alpha.6

  - All axis arguments must now be wrapped in newtype `Axis`.
  - Add method `.split_at(Axis, Ix)` to read-only and read-write array views.
  - Add constructors `ArrayView{,Mut}::from_slice` and array view methods
    are now visible in the docs.

- 0.4.0-alpha.5

  - Use new trait `LinalgScalar` for operations where we want type-based specialization.
    This shrinks the set of types that allow dot product, matrix multiply, mean.
  - Use BLAS acceleration transparently in `.dot()` (this is the first step).
  - Only OwnedArray and RcArray and not ArrayViewMut can now be used as consumed
    left hand operand for arithmetic operators. [See arithmetic operations docs!](
    https://docs.rs/ndarray/0.4.0-alpha.5/ndarray/struct.ArrayBase.html#arithmetic-operations)
  - Remove deprecated module `linalg` (it was already mostly empty)
  - Deprecate free function `zeros` in favour of static method `zeros`.

- 0.4.0-alpha.4

  - Rename `Array` to `RcArray`. Old name is deprecated.
  - Add methods `OuterIter::split_at`, `OuterIterMut::split_at`
  - Change `arr0, arr1, arr2, arr3` to return `OwnedArray`.
    Add `rcarr1, rcarr2, rcarr3` that return `RcArray`.

- 0.4.0-alpha.3

  - Improve arithmetic operations where the RHS is a broadcast 0-dimensional
    array.
  - Add read-only and read-write array views to the `rblas` integration.
    Added methods `AsBlas::{blas_view_checked, blas_view_mut_checked, bv, bvm}`.
  - Use hash_slice in `Hash` impl for arrays.

- 0.4.0-alpha.2

  - Add `ArrayBase::reversed_axes` which transposes an array.

- 0.4.0-alpha.1

  - Add checked and unchecked constructor methods for creating arrays
    from a vector and explicit dimension and stride, or with
    fortran (column major) memory order (marked `f`):
    
    + `ArrayBase::from_vec_dim`, `from_vec_dim_stride`,
      `from_vec_dim_stride_unchecked`,
    + `from_vec_dim_unchecked_f`, `from_elem_f`, `zeros_f`
    + View constructors `ArrayView::from_slice_dim_stride`,
      `ArrayViewMut::from_slice_dim_stride`.
    + Rename old `ArrayBase::from_vec_dim` to `from_vec_dim_unchecked`.

  - Check better for wraparound when computing the number of elements in a shape;
    this adds error cases that **panic** in `from_elem`, `zeros` etc,
    however *the new check will only ever panic in cases that would
    trigger debug assertions for overflow in the previous versions*!.
  - Add an array chunks iterator `.axis_chunks_iter()` and mutable version;
    it allows traversing the array in for example chunks of *n* rows at a time.
  - Remove methods and items deprecated since 0.3 or earlier; deprecated methods
    have notes about replacements in 0.3 docs.

- 0.3.1

  - Add `.row_mut()`, `.column_mut()`
  - Add `.axis_iter()`, `.axis_iter_mut()`

- **0.3.0**

  - Second round of API & consistency update is done
  - 0.3.0 highlight: **Index type** `Ix` **changed to** `usize`.
  - 0.3.0 highlight: Operator overloading for scalar and array arithmetic.
  - 0.3.0 highlight: Indexing with `a[[i, j, k]]` syntax.
  - Add `ArrayBase::eye(n)`
  - See below for more info

- 0.3.0-alpha.4

  - Shrink array view structs by removing their redundant slice field (see #45).
    Changed the definition of the view `type` aliases.
  - `.mat_mul()` and `.mat_mul_col()` now return `OwnedArray`.
    Use `.into_shared()` if you need an `Array`.
  - impl ExactSizeIterator where possible for iterators.
  - impl DoubleEndedIterator for `.outer_iter()` (and _mut).

- 0.3.0-alpha.3

  - `.subview()` changed to return an array view, also added `into_subview()`.
  - Add `.outer_iter()` and `.outer_iter_mut()` for iteration along the
    greatest axis of the array. Views also implement `into_outer_iter()` for
    “lifetime preserving” iterators.

- 0.3.0-alpha.2

  - Improve the strided last dimension case in `zip_mut_with` slightly
    (affects all binary operations).
  - Add `.row(i), .column(i)` for 2D arrays.
  - Deprecate `.row_iter(), .col_iter()`.
  - Add method `.dot()` for computing the dot product between two 1D arrays.


- 0.3.0-alpha.1

  - **Index type** `Ix` **changed to** `usize` (#9). Gives better iterator codegen
    and 64-bit size arrays.
  - Support scalar operands with arithmetic operators.
  - Change `.slice()` and `.diag()` to return array views, add `.into_diag()`.
  - Add ability to use fixed size arrays for array indexing, enabling syntax
    like `a[[i, j]]` for indexing.
  - Add `.ndim()`

- **0.2.0**

  - First chapter of API and performance evolution is done \\o/
  - 0.2.0 highlight: Vectorized (efficient) arithmetic operations
  - 0.2.0 highlight: Easier slicing using `s![]`
  - 0.2.0 highlight: Nicer API using views
  - 0.2.0 highlight: Bridging to BLAS functions.
  - See below for more info

- 0.2.0-alpha.9

  - Support strided matrices in `rblas` bridge, and fix a bug with
    non square matrices.
  - Deprecated all of module `linalg`.

- 0.2.0-alpha.8

  - **Note:** PACKAGE NAME CHANGED TO `ndarray`. Having package != crate ran
    into many quirks of various tools. Changing the package name is easier for
    everyone involved!
  - Optimized `scalar_sum()` so that it will vectorize for the floating point
    element case too.

- 0.2.0-alpha.7

  - Optimized arithmetic operations!

    - For c-contiguous arrays or arrays with c-contiguous lowest dimension
      they optimize very well, and can vectorize!

  - Add `.inner_iter()`, `.inner_iter_mut()`
  - Add `.fold()`, `.zip_mut_with()`
  - Add `.scalar_sum()`
  - Add example `examples/life.rs`

- 0.2.0-alpha.6

  - Add `#[deprecated]` attributes (enabled with new enough nightly)
  - Add `ArrayBase::linspace`, deprecate constructor `range`.

- 0.2.0-alpha.5

  - Add `s![...]`, a slice argument macro.
  - Add `aview_mut1()`, `zeros()`
  - Add `.diag_mut()` and deprecate `.diag_iter_mut()`, `.sub_iter_mut()`
  - Add `.uget()`, `.uget_mut()` for unchecked indexing and deprecate the
    old names.
  - Improve `ArrayBase::from_elem`
  - Removed `SliceRange`, replaced by `From` impls for `Si`.

- 0.2.0-alpha.4

  - Slicing methods like `.slice()` now take a fixed size array of `Si`
    as the slice description. This allows more type checking to verify that the
    number of axes is correct.
  - Add experimental `rblas` integration.
  - Add `into_shape()` which allows reshaping any array or view kind.

- 0.2.0-alpha.3

  - Add and edit a lot of documentation

- 0.2.0-alpha.2

  - Improve performance for iterators when the array data is in the default
    memory layout. The iterator then wraps the default slice iterator and
    loops will autovectorize.
  - Remove method `.indexed()` on iterators. Changed `Indexed` and added
    `ÌndexedMut`.
  - Added `.as_slice(), .as_mut_slice()`
  - Support rustc-serialize


- 0.2.0-alpha

  - Alpha release!
  - Introduce `ArrayBase`, `OwnedArray`, `ArrayView`, `ArrayViewMut`
  - All arithmetic operations should accept any array type
  - `Array` continues to refer to the default reference counted copy on write
    array
  - Add `.view()`, `.view_mut()`, `.to_owned()`, `.into_shared()`
  - Add `.slice_mut()`, `.subview_mut()`
  - Some operations now return `OwnedArray`:

    - `.map()`
    - `.sum()`
    - `.mean()`

  - Add `get`, `get_mut` to replace the now deprecated `at`, `at_mut`.
  - Fix bug in assign_scalar

- 0.1.1

  - Add Array::default
  - Fix bug in raw_data_mut

- 0.1.0

  - First release on crates.io
  - Starting point for evolution to come
 

[@adamreichold]: https://github.com/adamreichold
[@aganders3]: https://github.com/aganders3
[@bluss]: https://github.com/bluss
[@jturner314]: https://github.com/jturner314
[@LukeMathWalker]: https://github.com/LukeMathWalker
[@acj]: https://github.com/acj
[@adamreichold]: https://github.com/adamreichold
[@atouchet]: https://github.com/atouchet
[@andrei-papou]: https://github.com/andrei-papou
[@benkay]: https://github.com/benkay
[@cassiersg]: https://github.com/cassiersg
[@chohner]: https://github.com/chohner
[@dam5h]: https://github.com/dam5h
[@ethanhs]: https://github.com/ethanhs
[@d-dorazio]: https://github.com/d-dorazio
[@Eijebong]: https://github.com/Eijebong
[@HyeokSuLee]: https://github.com/HyeokSuLee
[@insideoutclub]: https://github.com/insideoutclub
[@JP-Ellis]: https://github.com/JP-Ellis
[@jimblandy]: https://github.com/jimblandy
[@LeSeulArtichaut]: https://github.com/LeSeulArtichaut
[@lifuyang]: https://github.com/liufuyang
[@kdubovikov]: https://github.com/kdubovikov
[@makotokato]: https://github.com/makotokato
[@max-sixty]: https://github.com/max-sixty
[@mneumann]: https://github.com/mneumann
[@mockersf]: https://github.com/mockersf
[@nilgoyette]: https://github.com/nilgoyette
[@nitsky]: https://github.com/nitsky
[@Rikorose]: https://github.com/Rikorose
[@rth]: https://github.com/rth
[@sebasv]: https://github.com/sebasv
[@SparrowLii]: https://github.com/SparrowLii
[@steffahn]: https://github.com/steffahn
[@stokhos]: https://github.com/stokhos
[@termoshtt]: https://github.com/termoshtt
[@TheLortex]: https://github.com/TheLortex
[@viniciusd]: https://github.com/viniciusd
[@VasanthakumarV]: https://github.com/VasanthakumarV
[@xd009642]: https://github.com/xd009642
[@Zuse64]: https://github.com/Zuse64
