# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

A [separate changelog is kept for rand_core](https://github.com/rust-random/core/blob/master/CHANGELOG.md).

You may also find the [Upgrade Guide](https://rust-random.github.io/book/update.html) useful.

## [0.10.3] — 2026-09-20

### Fixes
- Fix `WeightedIndex` panic when the sum of float weights is infinite; return `Error::Overflow` instead ([#1808])
- Fix spurious `Error::NonFinite` from `Uniform::new_inclusive` on large finite float ranges such as `0.0..=f64::MAX` ([#1821])
- Fix possible panic due to sampling a deserialized `Uniform<char>` ([#1831])

### Changes
- Report exact remaining lengths from `WeightedIndex::weights()` and reduce overhead when reading weights ([#1838])

[#1808]: https://github.com/rust-random/rand/pull/1808
[#1821]: https://github.com/rust-random/rand/pull/1821
[#1831]: https://github.com/rust-random/rand/pull/1831
[#1838]: https://github.com/rust-random/rand/pull/1838

## [0.10.2] — 2026-07-02

### Fixes
- Fix possible memory safety violation due to deserialization of `UniformChar` from bad source ([#1790])

### Changes
- Document required output order of fn `partial_shuffle` and apply `#[must_use]` ([#1769])
- Avoid usage of `unsafe` in contexts where non-local memory corruption could invalidate contract ([#1791])

[#1769]: https://github.com/rust-random/rand/pull/1769
[#1790]: https://github.com/rust-random/rand/pull/1790
[#1791]: https://github.com/rust-random/rand/pull/1791

## [0.10.1] — 2026-02-11
This release includes a fix for a soundness bug; see [#1763].

### Changes
- Document panic behavior of `make_rng` and add `#[track_caller]` ([#1761])
- Deprecate feature `log` ([#1763])

[#1761]: https://github.com/rust-random/rand/pull/1761
[#1763]: https://github.com/rust-random/rand/pull/1763

## [0.10.0] - 2026-02-08

### Changes
- The dependency on `rand_chacha` has been replaced with a dependency on `chacha20`. This changes the implementation behind `StdRng`, but the output remains the same. There may be some API breakage when using the ChaCha-types directly as these are now the ones in `chacha20` instead of `rand_chacha` ([#1642]).
- Rename fns `IndexedRandom::choose_multiple` -> `sample`, `choose_multiple_array` -> `sample_array`, `choose_multiple_weighted` -> `sample_weighted`, struct `SliceChooseIter` -> `IndexedSamples` and fns `IteratorRandom::choose_multiple` -> `sample`, `choose_multiple_fill` -> `sample_fill` ([#1632])
- Use Edition 2024 and MSRV 1.85 ([#1653])
- Let `Fill` be implemented for element types, not sliceable types ([#1652])
- Fix `OsError::raw_os_error` on UEFI targets by returning `Option<usize>` ([#1665])
- Replace fn `TryRngCore::read_adapter(..) -> RngReadAdapter` with simpler struct `RngReader` ([#1669])
- Remove fns `SeedableRng::from_os_rng`, `try_from_os_rng` ([#1674])
- Remove `Clone` support for `StdRng`, `ReseedingRng` ([#1677])
- Use `postcard` instead of `bincode` to test the serde feature ([#1693])
- Avoid excessive allocation in `IteratorRandom::sample` when `amount` is much larger than iterator size ([#1695])
- Rename `os_rng` -> `sys_rng`, `OsRng` -> `SysRng`, `OsError` -> `SysError` ([#1697])
- Rename `Rng` -> `RngExt` as upstream `rand_core` has renamed `RngCore` -> `Rng` ([#1717])

### Additions
- Add fns `IndexedRandom::choose_iter`, `choose_weighted_iter` ([#1632])
- Pub export `Xoshiro128PlusPlus`, `Xoshiro256PlusPlus` prngs ([#1649])
- Pub export `ChaCha8Rng`, `ChaCha12Rng`, `ChaCha20Rng` behind `chacha` feature ([#1659])
- Fn `rand::make_rng() -> R where R: SeedableRng` ([#1734])

### Removals
- Removed `ReseedingRng` ([#1722])
- Removed unused feature "nightly" ([#1732])
- Removed feature `small_rng` ([#1732])

[#1632]: https://github.com/rust-random/rand/pull/1632
[#1642]: https://github.com/rust-random/rand/pull/1642
[#1649]: https://github.com/rust-random/rand/pull/1649
[#1652]: https://github.com/rust-random/rand/pull/1652
[#1653]: https://github.com/rust-random/rand/pull/1653
[#1659]: https://github.com/rust-random/rand/pull/1659
[#1665]: https://github.com/rust-random/rand/pull/1665
[#1669]: https://github.com/rust-random/rand/pull/1669
[#1674]: https://github.com/rust-random/rand/pull/1674
[#1677]: https://github.com/rust-random/rand/pull/1677
[#1693]: https://github.com/rust-random/rand/pull/1693
[#1695]: https://github.com/rust-random/rand/pull/1695
[#1697]: https://github.com/rust-random/rand/pull/1697
[#1717]: https://github.com/rust-random/rand/pull/1717
[#1722]: https://github.com/rust-random/rand/pull/1722
[#1732]: https://github.com/rust-random/rand/pull/1732
[#1734]: https://github.com/rust-random/rand/pull/1734

## [0.9.2] - 2025-07-20

### Deprecated
- Deprecate `rand::rngs::mock` module and `StepRng` generator ([#1634])

### Additions
- Enable `WeightedIndex<usize>` (de)serialization ([#1646])

## [0.9.1] - 2025-04-17

### Security and unsafe
- Revise "not a crypto library" policy again ([#1565])
- Remove `zerocopy` dependency from `rand` ([#1579])

### Fixes
- Fix feature `simd_support` for recent nightly rust ([#1586])

### Changes
- Allow `fn rand::seq::index::sample_weighted` and `fn IndexedRandom::choose_multiple_weighted` to return fewer than `amount` results ([#1623]), reverting an undocumented change ([#1382]) to the previous release.

### Additions
- Add `rand::distr::Alphabetic` distribution. ([#1587])
- Re-export `rand_core` ([#1604])

[#1565]: https://github.com/rust-random/rand/pull/1565
[#1579]: https://github.com/rust-random/rand/pull/1579
[#1586]: https://github.com/rust-random/rand/pull/1586
[#1587]: https://github.com/rust-random/rand/pull/1587
[#1604]: https://github.com/rust-random/rand/pull/1604
[#1623]: https://github.com/rust-random/rand/pull/1623
[#1634]: https://github.com/rust-random/rand/pull/1634
[#1646]: https://github.com/rust-random/rand/pull/1646

## [0.9.0] - 2025-01-27

### Security and unsafe
- Policy: "rand is not a crypto library" ([#1514])
- Remove fork-protection from `ReseedingRng` and `ThreadRng`. Instead, it is recommended to call `ThreadRng::reseed` on fork. ([#1379])
- Use `zerocopy` to replace some `unsafe` code ([#1349], [#1393], [#1446], [#1502])

### Dependencies
- Bump the MSRV to 1.63.0 ([#1207], [#1246], [#1269], [#1341], [#1416], [#1536]); note that 1.60.0 may work for dependents when using `--ignore-rust-version`
- Update to `rand_core` v0.9.0 ([#1558])

### Features
- Support `std` feature without `getrandom` or `rand_chacha` ([#1354])
- Enable feature `small_rng` by default ([#1455])
- Remove implicit feature `rand_chacha`; use `std_rng` instead. ([#1473])
- Rename feature `serde1` to `serde` ([#1477])
- Rename feature `getrandom` to `os_rng` ([#1537])
- Add feature `thread_rng` ([#1547])

### API changes: rand_core traits
- Add fn `RngCore::read_adapter` implementing `std::io::Read` ([#1267])
- Add trait `CryptoBlockRng: BlockRngCore`; make `trait CryptoRng: RngCore` ([#1273])
- Add traits `TryRngCore`, `TryCryptoRng` ([#1424], [#1499])
- Rename `fn SeedableRng::from_rng` -> `try_from_rng` and add infallible variant `fn from_rng` ([#1424])
- Rename `fn SeedableRng::from_entropy` -> `from_os_rng` and add fallible variant `fn try_from_os_rng` ([#1424])
- Add bounds `Clone` and `AsRef` to associated type `SeedableRng::Seed` ([#1491])

### API changes: Rng trait and top-level fns
- Rename fn `rand::thread_rng()` to `rand::rng()` and remove from the prelude ([#1506])
- Remove fn `rand::random()` from the prelude ([#1506])
- Add top-level fns `random_iter`, `random_range`, `random_bool`, `random_ratio`, `fill` ([#1488])
- Re-introduce fn `Rng::gen_iter` as `random_iter` ([#1305], [#1500])
- Rename fn `Rng::gen` to `random` to avoid conflict with the new `gen` keyword in Rust 2024 ([#1438])
- Rename fns `Rng::gen_range` to `random_range`, `gen_bool` to `random_bool`, `gen_ratio` to `random_ratio` ([#1505])
- Annotate panicking methods with `#[track_caller]` ([#1442], [#1447])

### API changes: RNGs
- Fix `<SmallRng as SeedableRng>::Seed` size to 256 bits ([#1455])
- Remove first parameter (`rng`) of `ReseedingRng::new` ([#1533])

### API changes: Sequences
- Split trait `SliceRandom` into `IndexedRandom`, `IndexedMutRandom`, `SliceRandom` ([#1382])
- Add `IndexedRandom::choose_multiple_array`, `index::sample_array` ([#1453], [#1469])

### API changes: Distributions: renames
- Rename module `rand::distributions` to `rand::distr` ([#1470])
- Rename distribution `Standard` to `StandardUniform` ([#1526])
- Move `distr::Slice` -> `distr::slice::Choose`, `distr::EmptySlice` -> `distr::slice::Empty` ([#1548])
- Rename trait `distr::DistString` -> `distr::SampleString` ([#1548])
- Rename `distr::DistIter` -> `distr::Iter`, `distr::DistMap` -> `distr::Map` ([#1548])

### API changes: Distributions
- Relax `Sized` bound on `Distribution<T> for &D` ([#1278])
- Remove impl of `Distribution<Option<T>>` for `StandardUniform` ([#1526])
- Let distribution `StandardUniform` support all `NonZero*` types ([#1332])
- Fns `{Uniform, UniformSampler}::{new, new_inclusive}` return a `Result` (instead of potentially panicking) ([#1229])
- Distribution `Uniform` implements `TryFrom` instead of `From` for ranges ([#1229])
- Add `UniformUsize` ([#1487])
- Remove support for generating `isize` and `usize` values with `StandardUniform`, `Uniform` (except via `UniformUsize`) and `Fill` and usage as a `WeightedAliasIndex` weight ([#1487])
- Add impl `DistString` for distributions `Slice<char>` and `Uniform<char>` ([#1315])
- Add fn `Slice::num_choices` ([#1402])
- Add fn `p()` for distribution `Bernoulli` to access probability ([#1481])

### API changes: Weighted distributions
- Add `pub` module `rand::distr::weighted`, moving `WeightedIndex` there ([#1548])
- Add trait `weighted::Weight`, allowing `WeightedIndex` to trap overflow ([#1353])
- Add fns `weight, weights, total_weight` to distribution `WeightedIndex` ([#1420])
- Rename enum `WeightedError` to `weighted::Error`, revising variants ([#1382]) and mark as `#[non_exhaustive]` ([#1480])

### API changes: SIMD
- Switch to `std::simd`, expand SIMD & docs ([#1239])

### Reproducibility-breaking changes
- Make `ReseedingRng::reseed` discard remaining data from the last block generated ([#1379])
- Change fn `SmallRng::seed_from_u64` implementation ([#1203])
- Allow `UniformFloat::new` samples and `UniformFloat::sample_single` to yield `high` ([#1462])
- Fix portability of distribution `Slice` ([#1469])
- Make `Uniform` for `usize` portable via `UniformUsize` ([#1487])
- Fix `IndexdRandom::choose_multiple_weighted` for very small seeds and optimize for large input length / low memory ([#1530])

### Reproducibility-breaking optimisations
- Optimize fn `sample_floyd`, affecting output of `rand::seq::index::sample` and `rand::seq::SliceRandom::choose_multiple` ([#1277])
- New, faster algorithms for `IteratorRandom::choose` and `choose_stable` ([#1268])
- New, faster algorithms for `SliceRandom::shuffle` and `partial_shuffle` ([#1272])
- Optimize distribution `Uniform`: use Canon's method (single sampling) / Lemire's method (distribution sampling) for faster sampling (breaks value stability; [#1287])
- Optimize fn `sample_single_inclusive` for floats (+~20% perf) ([#1289])

### Other optimisations
- Improve `SmallRng` initialization performance ([#1482])
- Optimise SIMD widening multiply ([#1247])

### Other
- Add `Cargo.lock.msrv` file ([#1275])
- Reformat with `rustfmt` and enforce ([#1448])
- Apply Clippy suggestions and enforce ([#1448], [#1474])
- Move all benchmarks to new `benches` crate ([#1329], [#1439]) and migrate to Criterion ([#1490])

### Documentation
- Improve `ThreadRng` related docs ([#1257])
- Docs: enable experimental `--generate-link-to-definition` feature ([#1327])
- Better doc of crate features, use `doc_auto_cfg` ([#1411], [#1450])

[#1203]: https://github.com/rust-random/rand/pull/1203
[#1207]: https://github.com/rust-random/rand/pull/1207
[#1229]: https://github.com/rust-random/rand/pull/1229
[#1239]: https://github.com/rust-random/rand/pull/1239
[#1246]: https://github.com/rust-random/rand/pull/1246
[#1247]: https://github.com/rust-random/rand/pull/1247
[#1257]: https://github.com/rust-random/rand/pull/1257
[#1267]: https://github.com/rust-random/rand/pull/1267
[#1268]: https://github.com/rust-random/rand/pull/1268
[#1269]: https://github.com/rust-random/rand/pull/1269
[#1272]: https://github.com/rust-random/rand/pull/1272
[#1273]: https://github.com/rust-random/rand/pull/1273
[#1275]: https://github.com/rust-random/rand/pull/1275
[#1277]: https://github.com/rust-random/rand/pull/1277
[#1278]: https://github.com/rust-random/rand/pull/1278
[#1287]: https://github.com/rust-random/rand/pull/1287
[#1289]: https://github.com/rust-random/rand/pull/1289
[#1305]: https://github.com/rust-random/rand/pull/1305
[#1315]: https://github.com/rust-random/rand/pull/1315
[#1327]: https://github.com/rust-random/rand/pull/1327
[#1329]: https://github.com/rust-random/rand/pull/1329
[#1332]: https://github.com/rust-random/rand/pull/1332
[#1341]: https://github.com/rust-random/rand/pull/1341
[#1349]: https://github.com/rust-random/rand/pull/1349
[#1353]: https://github.com/rust-random/rand/pull/1353
[#1354]: https://github.com/rust-random/rand/pull/1354
[#1379]: https://github.com/rust-random/rand/pull/1379
[#1382]: https://github.com/rust-random/rand/pull/1382
[#1393]: https://github.com/rust-random/rand/pull/1393
[#1402]: https://github.com/rust-random/rand/pull/1402
[#1411]: https://github.com/rust-random/rand/pull/1411
[#1416]: https://github.com/rust-random/rand/pull/1416
[#1420]: https://github.com/rust-random/rand/pull/1420
[#1424]: https://github.com/rust-random/rand/pull/1424
[#1438]: https://github.com/rust-random/rand/pull/1438
[#1439]: https://github.com/rust-random/rand/pull/1439
[#1442]: https://github.com/rust-random/rand/pull/1442
[#1446]: https://github.com/rust-random/rand/pull/1446
[#1447]: https://github.com/rust-random/rand/pull/1447
[#1448]: https://github.com/rust-random/rand/pull/1448
[#1450]: https://github.com/rust-random/rand/pull/1450
[#1453]: https://github.com/rust-random/rand/pull/1453
[#1455]: https://github.com/rust-random/rand/pull/1455
[#1462]: https://github.com/rust-random/rand/pull/1462
[#1469]: https://github.com/rust-random/rand/pull/1469
[#1470]: https://github.com/rust-random/rand/pull/1470
[#1473]: https://github.com/rust-random/rand/pull/1473
[#1474]: https://github.com/rust-random/rand/pull/1474
[#1477]: https://github.com/rust-random/rand/pull/1477
[#1480]: https://github.com/rust-random/rand/pull/1480
[#1481]: https://github.com/rust-random/rand/pull/1481
[#1482]: https://github.com/rust-random/rand/pull/1482
[#1487]: https://github.com/rust-random/rand/pull/1487
[#1488]: https://github.com/rust-random/rand/pull/1488
[#1490]: https://github.com/rust-random/rand/pull/1490
[#1491]: https://github.com/rust-random/rand/pull/1491
[#1499]: https://github.com/rust-random/rand/pull/1499
[#1500]: https://github.com/rust-random/rand/pull/1500
[#1502]: https://github.com/rust-random/rand/pull/1502
[#1505]: https://github.com/rust-random/rand/pull/1505
[#1506]: https://github.com/rust-random/rand/pull/1506
[#1514]: https://github.com/rust-random/rand/pull/1514
[#1526]: https://github.com/rust-random/rand/pull/1526
[#1530]: https://github.com/rust-random/rand/pull/1530
[#1533]: https://github.com/rust-random/rand/pull/1533
[#1536]: https://github.com/rust-random/rand/pull/1536
[#1537]: https://github.com/rust-random/rand/pull/1537
[#1547]: https://github.com/rust-random/rand/pull/1547
[#1548]: https://github.com/rust-random/rand/pull/1548
[#1558]: https://github.com/rust-random/rand/pull/1558

## [0.8.5] - 2021-08-20

### Fixes
- Fix build on non-32/64-bit architectures ([#1144])
- Fix "min_const_gen" feature for `no_std` ([#1173])
- Check `libc::pthread_atfork` return value with panic on error ([#1178])
- More robust reseeding in case `ReseedingRng` is used from a fork handler ([#1178])
- Fix nightly: remove unused `slice_partition_at_index` feature ([#1215])
- Fix nightly + `simd_support`: update `packed_simd` ([#1216])

### Rngs
- `StdRng`: Switch from HC128 to ChaCha12 on emscripten ([#1142]).
  We now use ChaCha12 on all platforms.

### Documentation
- Added docs about rand's use of const generics ([#1150])
- Better random chars example ([#1157])

[#1142]: https://github.com/rust-random/rand/pull/1142
[#1144]: https://github.com/rust-random/rand/pull/1144
[#1150]: https://github.com/rust-random/rand/pull/1150
[#1157]: https://github.com/rust-random/rand/pull/1157
[#1173]: https://github.com/rust-random/rand/pull/1173
[#1178]: https://github.com/rust-random/rand/pull/1178
[#1215]: https://github.com/rust-random/rand/pull/1215
[#1216]: https://github.com/rust-random/rand/pull/1216

## [0.8.4] - 2021-06-15

### Additions
- Use const-generics to support arrays of all sizes ([#1104])
- Implement `Clone` and `Copy` for `Alphanumeric` ([#1126])
- Add `Distribution::map` to derive a distribution using a closure ([#1129])
- Add `Slice` distribution ([#1107])
- Add `DistString` trait with impls for `Standard` and `Alphanumeric` ([#1133])

### Other
- Reorder asserts in `Uniform` float distributions for easier debugging of non-finite arguments ([#1094], [#1108])
- Add range overflow check in `Uniform` float distributions ([#1108])
- Deprecate `rngs::adapter::ReadRng` ([#1130])

## [0.8.3] - 2021-01-25

### Fixes
- Fix `no-std` + `alloc` build by gating `choose_multiple_weighted` on `std` ([#1088])

## [0.8.2] - 2021-01-12

### Fixes
- Fix panic in `UniformInt::sample_single_inclusive` and `Rng::gen_range` when providing a full integer range (eg `0..=MAX`) ([#1087])

## [0.8.1] - 2020-12-31

### Other
- Enable all stable features in the playground ([#1081])
## [0.8.0] - 2020-12-18

### Platform support
- The minimum supported Rust version is now 1.36 ([#1011])
- `getrandom` updated to v0.2 ([#1041])
- Remove `wasm-bindgen` and `stdweb` feature flags. For details of WASM support, see the [getrandom documentation](https://docs.rs/getrandom/latest). ([#948])
- `ReadRng::next_u32` and `next_u64` now use little-Endian conversion instead of native-Endian, affecting results on Big-Endian platforms ([#1061])
- The `nightly` feature no longer implies the `simd_support` feature ([#1048])
- Fix `simd_support` feature to work on current nightlies ([#1056])

### Rngs
- `ThreadRng` is no longer `Copy` to enable safe usage within thread-local destructors ([#1035])
- `gen_range(a, b)` was replaced with `gen_range(a..b)`. `gen_range(a..=b)` is also supported. Note that `a` and `b` can no longer be references or SIMD types. ([#744], [#1003])
- Replace `AsByteSliceMut` with `Fill` and add support for `[bool], [char], [f32], [f64]` ([#940])
- Restrict `rand::rngs::adapter` to `std` ([#1027]; see also [#928])
- `StdRng`: add new `std_rng` feature flag (enabled by default, but might need to be used if disabling default crate features) ([#948])
- `StdRng`: Switch from ChaCha20 to ChaCha12 for better performance ([#1028])
- `SmallRng`: Replace PCG algorithm with xoshiro{128,256}++ ([#1038])

### Sequences
- Add `IteratorRandom::choose_stable` as an alternative to `choose` which does not depend on size hints ([#1057])
- Improve accuracy and performance of `IteratorRandom::choose` ([#1059])
- Implement `IntoIterator` for `IndexVec`, replacing the `into_iter` method ([#1007])
- Add value stability tests for `seq` module ([#933])

### Misc
- Support `PartialEq` and `Eq` for `StdRng`, `SmallRng` and `StepRng` ([#979])
- Added a `serde1` feature and added Serialize/Deserialize to `UniformInt` and `WeightedIndex` ([#974])
- Drop some unsafe code ([#962], [#963], [#1011])
- Reduce packaged crate size ([#983])
- Migrate to GitHub Actions from Travis+AppVeyor ([#1073])

### Distributions
- `Alphanumeric` samples bytes instead of chars ([#935])
- `Uniform` now supports `char`, enabling `rng.gen_range('A'..='Z')` ([#1068])
- Add `UniformSampler::sample_single_inclusive` ([#1003])

#### Weighted sampling
- Implement weighted sampling without replacement ([#976], [#1013])
- `rand::distributions::alias_method::WeightedIndex` was moved to `rand_distr::WeightedAliasIndex`. The simpler alternative `rand::distribution::WeightedIndex` remains. ([#945])
- Improve treatment of rounding errors in `WeightedIndex::update_weights` ([#956])
- `WeightedIndex`: return error on NaN instead of panic ([#1005])

### Documentation
- Document types supported by `random` ([#994])
- Document notes on password generation ([#995])
- Note that `SmallRng` may not be the best choice for performance and in some other cases ([#1038])
- Use `doc(cfg)` to annotate feature-gated items ([#1019])
- Adjust README ([#1065])

[#744]: https://github.com/rust-random/rand/pull/744
[#933]: https://github.com/rust-random/rand/pull/933
[#935]: https://github.com/rust-random/rand/pull/935
[#940]: https://github.com/rust-random/rand/pull/940
[#945]: https://github.com/rust-random/rand/pull/945
[#948]: https://github.com/rust-random/rand/pull/948
[#956]: https://github.com/rust-random/rand/pull/956
[#962]: https://github.com/rust-random/rand/pull/962
[#963]: https://github.com/rust-random/rand/pull/963
[#974]: https://github.com/rust-random/rand/pull/974
[#976]: https://github.com/rust-random/rand/pull/976
[#979]: https://github.com/rust-random/rand/pull/979
[#983]: https://github.com/rust-random/rand/pull/983
[#994]: https://github.com/rust-random/rand/pull/994
[#995]: https://github.com/rust-random/rand/pull/995
[#1003]: https://github.com/rust-random/rand/pull/1003
[#1005]: https://github.com/rust-random/rand/pull/1005
[#1007]: https://github.com/rust-random/rand/pull/1007
[#1011]: https://github.com/rust-random/rand/pull/1011
[#1013]: https://github.com/rust-random/rand/pull/1013
[#1019]: https://github.com/rust-random/rand/pull/1019
[#1027]: https://github.com/rust-random/rand/pull/1027
[#1028]: https://github.com/rust-random/rand/pull/1028
[#1035]: https://github.com/rust-random/rand/pull/1035
[#1038]: https://github.com/rust-random/rand/pull/1038
[#1041]: https://github.com/rust-random/rand/pull/1041
[#1048]: https://github.com/rust-random/rand/pull/1048
[#1056]: https://github.com/rust-random/rand/pull/1056
[#1057]: https://github.com/rust-random/rand/pull/1057
[#1059]: https://github.com/rust-random/rand/pull/1059
[#1061]: https://github.com/rust-random/rand/pull/1061
[#1065]: https://github.com/rust-random/rand/pull/1065
[#1068]: https://github.com/rust-random/rand/pull/1068
[#1073]: https://github.com/rust-random/rand/pull/1073

## [0.7.3] - 2020-01-10

### Fixes
- The `Bernoulli` distribution constructors now reports an error on NaN and on `denominator == 0`. ([#925])
- Use `std::sync::Once` to register fork handler, avoiding possible atomicity violation ([#928])
- Fix documentation on the precision of generated floating-point values

### Changes
- Unix: make libc dependency optional; only use fork protection with std feature ([#928])

### Additions
- Implement `std::error::Error` for `BernoulliError` ([#919])

## [0.7.2] - 2019-09-16

### Fixes
- Fix dependency on `rand_core` 0.5.1 ([#890])

### Additions
- Unit tests for value stability of distributions added ([#888])

## [0.7.1] - 2019-09-13

### Yanked
This release was yanked since it depends on `rand_core::OsRng` added in 0.5.1 but specifies a dependency on version 0.5.0 ([#890]), causing broken builds when updating from `rand 0.7.0` without also updating `rand_core`.

### Fixes
- Fix `no_std` behaviour, appropriately enable c2-chacha's `std` feature ([#844])
- `alloc` feature in `no_std` is available since Rust 1.36 ([#856])
- Fix or squelch issues from Clippy lints ([#840])

### Additions
- Add a `no_std` target to CI to continuously evaluate `no_std` status ([#844])
- `WeightedIndex`: allow adjusting a sub-set of weights ([#866])

## [0.7.0] - 2019-06-28

### Fixes
- Fix incorrect pointer usages revealed by Miri testing ([#780], [#781])
- Fix (tiny!) bias in `Uniform` for 8- and 16-bit ints ([#809])

### Crate
- Bumped MSRV (min supported Rust version) to 1.32.0
- Updated to Rust Edition 2018 ([#823], [#824])
- Removed dependence on `rand_xorshift`, `rand_isaac`, `rand_jitter` crates ([#759], [#765])
- Remove dependency on `winapi` ([#724])
- Removed all `build.rs` files ([#824])
- Removed code already deprecated in version 0.6 ([#757])
- Removed the serde1 feature (It's still available for backwards compatibility, but it does not do anything. [#830])
- Many documentation changes

### rand_core
- Updated to `rand_core` 0.5.0
- `Error` type redesigned with new API ([#800])
- Move `from_entropy` method to `SeedableRng` and remove `FromEntropy` ([#800])
- `SeedableRng::from_rng` is now expected to be value-stable ([#815])

### Standard RNGs
- OS interface moved from `rand_os` to new `getrandom` crate ([#765], [getrandom](https://github.com/rust-random/getrandom))
- Use ChaCha for `StdRng` and `ThreadRng` ([#792])
- Feature-gate `SmallRng` ([#792])
- `ThreadRng` now supports `Copy` ([#758])
- Deprecated `EntropyRng` ([#765])
- Enable fork protection of `ReseedingRng` without `std` ([#724])

### Distributions
- Many distributions have been moved to `rand_distr` ([#761])
- `Bernoulli::new` constructor now returns a `Result` ([#803])
- `Distribution::sample_iter` adjusted for more flexibility ([#758])
- Added `distributions::weighted::alias_method::WeightedIndex` for `O(1)` sampling ([#692])
- Support sampling `NonZeroU*` types with the `Standard` distribution ([#728])
- Optimised `Binomial` distribution sampling ([#735], [#740], [#752])
- Optimised SIMD float sampling ([#739])

### Sequences
- Make results portable across 32- and 64-bit by using `u32` samples for `usize` where possible ([#809])

[#692]: https://github.com/rust-random/rand/pull/692
[#724]: https://github.com/rust-random/rand/pull/724
[#728]: https://github.com/rust-random/rand/pull/728
[#735]: https://github.com/rust-random/rand/pull/735
[#739]: https://github.com/rust-random/rand/pull/739
[#740]: https://github.com/rust-random/rand/pull/740
[#752]: https://github.com/rust-random/rand/pull/752
[#757]: https://github.com/rust-random/rand/pull/757
[#758]: https://github.com/rust-random/rand/pull/758
[#759]: https://github.com/rust-random/rand/pull/759
[#761]: https://github.com/rust-random/rand/pull/761
[#765]: https://github.com/rust-random/rand/pull/765
[#780]: https://github.com/rust-random/rand/pull/780
[#781]: https://github.com/rust-random/rand/pull/781
[#792]: https://github.com/rust-random/rand/pull/792
[#800]: https://github.com/rust-random/rand/pull/800
[#803]: https://github.com/rust-random/rand/pull/803
[#809]: https://github.com/rust-random/rand/pull/809
[#815]: https://github.com/rust-random/rand/pull/815
[#823]: https://github.com/rust-random/rand/pull/823
[#824]: https://github.com/rust-random/rand/pull/824
[#830]: https://github.com/rust-random/rand/pull/830
[#840]: https://github.com/rust-random/rand/pull/840
[#844]: https://github.com/rust-random/rand/pull/844
[#856]: https://github.com/rust-random/rand/pull/856
[#866]: https://github.com/rust-random/rand/pull/866
[#888]: https://github.com/rust-random/rand/pull/888
[#890]: https://github.com/rust-random/rand/pull/890
[#919]: https://github.com/rust-random/rand/pull/919
[#925]: https://github.com/rust-random/rand/pull/925
[#928]: https://github.com/rust-random/rand/pull/928

## [0.6.5] - 2019-01-28

### Crates
- Update `rand_core` to 0.4 ([#703])
- Move `JitterRng` to its own crate ([#685])
- Add a wasm-bindgen test crate ([#696])

### Platforms
- Fuchsia: Replaced fuchsia-zircon with fuchsia-cprng

### Doc
- Use RFC 1946 for doc links ([#691])
- Fix some doc links and notes ([#711])

## [0.6.4] - 2019-01-08

### Fixes
- Move wasm-bindgen shims to correct crate ([#686])
- Make `wasm32-unknown-unknown` compile but fail at run-time if missing bindings ([#686])

## [0.6.3] - 2019-01-04

### Fixes
- Make the `std` feature require the optional `rand_os` dependency ([#675])
- Re-export the optional WASM dependencies of `rand_os` from `rand` to avoid breakage ([#674])

## [0.6.2] - 2019-01-04

### Additions
- Add `Default` for `ThreadRng` ([#657])
- Move `rngs::OsRng` to `rand_os` sub-crate; clean up code; use as dependency ([#643])
- Add `rand_xoshiro` sub-crate, plus benchmarks ([#642], [#668])

### Fixes
- Fix bias in `UniformInt::sample_single` ([#662])
- Use `autocfg` instead of `rustc_version` for rustc version detection ([#664])
- Disable `i128` and `u128` if the `target_os` is `emscripten` ([#671]: work-around Emscripten limitation)
- CI fixes ([#660], [#671])

### Optimisations
- Optimise memory usage of `UnitCircle` and `UnitSphereSurface` distributions (no PR)

## [0.6.1] - 2018-11-22
- Support sampling `Duration` also for `no_std` (only since Rust 1.25) ([#649])
- Disable default features of `libc` ([#647])

## [0.6.0] - 2018-11-14

### Project organisation
- Rand has moved from [rust-lang-nursery](https://github.com/rust-lang-nursery/rand) to [rust-random](https://github.com/rust-random/rand)! ([#578])
- Created [The Rust Random Book](https://rust-random.github.io/book/) ([source](https://github.com/rust-random/book))
- Update copyright and licence notices ([#591], [#611])
- Migrate policy documentation from the wiki ([#544])

### Platforms
- Add fork protection on Unix ([#466])
- Added support for wasm-bindgen ([#541], [#559], [#562], [#600])
- Enable `OsRng` for powerpc64, sparc and sparc64 ([#609])
- Use `syscall` from `libc` on Linux instead of redefining it ([#629])

### RNGs
- Switch `SmallRng` to use PCG ([#623])
- Implement `Pcg32` and `Pcg64Mcg` generators ([#632])
- Move ISAAC RNGs to a dedicated crate ([#551])
- Move Xorshift RNG to its own crate ([#557])
- Move ChaCha and HC128 RNGs to dedicated crates ([#607], [#636])
- Remove usage of `Rc` from `ThreadRng` ([#615])

### Sampling and distributions
- Implement `Rng.gen_ratio()` and `Bernoulli::new_ratio()` ([#491])
- Make `Uniform` strictly respect `f32` / `f64` high/low bounds ([#477])
- Allow `gen_range` and `Uniform` to work on non-`Copy` types ([#506])
- `Uniform` supports inclusive ranges: `Uniform::from(a..=b)`. This is automatically enabled for Rust >= 1.27. ([#566])
- Implement `TrustedLen` and `FusedIterator` for `DistIter` ([#620])

#### New distributions
- Add the `Dirichlet` distribution ([#485])
- Added sampling from the unit sphere and circle ([#567])
- Implement the triangular distribution ([#575])
- Implement the Weibull distribution ([#576])
- Implement the Beta distribution ([#574])

#### Optimisations
- Optimise `Bernoulli::new` ([#500])
- Optimise `char` sampling ([#519])
- Optimise sampling of `std::time::Duration` ([#583])

### Sequences
- Redesign the `seq` module ([#483], [#515])
- Add `WeightedIndex` and `choose_weighted` ([#518], [#547])
- Optimised and changed return type of the `sample_indices` function ([#479])
- Use `Iterator::size_hint()` to speed up `IteratorRandom::choose` ([#593])

### SIMD
- Support for generating SIMD types ([#523], [#542], [#561], [#630])

### Other
- Revise CI scripts ([#632], [#635])
- Remove functionality already deprecated in 0.5 ([#499])
- Support for `i128` and `u128` is automatically enabled for Rust >= 1.26. This renders the `i128_support` feature obsolete. It still exists for backwards compatibility but does not have any effect. This breaks programs using Rand with `i128_support` on nightlies older than Rust 1.26. ([#571])

[#466]: https://github.com/rust-random/rand/pull/466
[#477]: https://github.com/rust-random/rand/pull/477
[#479]: https://github.com/rust-random/rand/pull/479
[#483]: https://github.com/rust-random/rand/pull/483
[#485]: https://github.com/rust-random/rand/pull/485
[#491]: https://github.com/rust-random/rand/pull/491
[#499]: https://github.com/rust-random/rand/pull/499
[#500]: https://github.com/rust-random/rand/pull/500
[#506]: https://github.com/rust-random/rand/pull/506
[#515]: https://github.com/rust-random/rand/pull/515
[#518]: https://github.com/rust-random/rand/pull/518
[#519]: https://github.com/rust-random/rand/pull/519
[#523]: https://github.com/rust-random/rand/pull/523
[#541]: https://github.com/rust-random/rand/pull/541
[#542]: https://github.com/rust-random/rand/pull/542
[#544]: https://github.com/rust-random/rand/pull/544
[#547]: https://github.com/rust-random/rand/pull/547
[#551]: https://github.com/rust-random/rand/pull/551
[#557]: https://github.com/rust-random/rand/pull/557
[#559]: https://github.com/rust-random/rand/pull/559
[#561]: https://github.com/rust-random/rand/pull/561
[#562]: https://github.com/rust-random/rand/pull/562
[#566]: https://github.com/rust-random/rand/pull/566
[#567]: https://github.com/rust-random/rand/pull/567
[#571]: https://github.com/rust-random/rand/pull/571
[#574]: https://github.com/rust-random/rand/pull/574
[#575]: https://github.com/rust-random/rand/pull/575
[#576]: https://github.com/rust-random/rand/pull/576
[#578]: https://github.com/rust-random/rand/pull/578
[#583]: https://github.com/rust-random/rand/pull/583
[#591]: https://github.com/rust-random/rand/pull/591
[#593]: https://github.com/rust-random/rand/pull/593
[#600]: https://github.com/rust-random/rand/pull/600
[#607]: https://github.com/rust-random/rand/pull/607
[#609]: https://github.com/rust-random/rand/pull/609
[#611]: https://github.com/rust-random/rand/pull/611
[#615]: https://github.com/rust-random/rand/pull/615
[#620]: https://github.com/rust-random/rand/pull/620
[#623]: https://github.com/rust-random/rand/pull/623
[#629]: https://github.com/rust-random/rand/pull/629
[#630]: https://github.com/rust-random/rand/pull/630
[#632]: https://github.com/rust-random/rand/pull/632
[#635]: https://github.com/rust-random/rand/pull/635
[#636]: https://github.com/rust-random/rand/pull/636
[#642]: https://github.com/rust-random/rand/pull/642
[#643]: https://github.com/rust-random/rand/pull/643
[#647]: https://github.com/rust-random/rand/pull/647
[#649]: https://github.com/rust-random/rand/pull/649
[#657]: https://github.com/rust-random/rand/pull/657
[#660]: https://github.com/rust-random/rand/pull/660
[#662]: https://github.com/rust-random/rand/pull/662
[#664]: https://github.com/rust-random/rand/pull/664
[#668]: https://github.com/rust-random/rand/pull/668
[#671]: https://github.com/rust-random/rand/pull/671
[#674]: https://github.com/rust-random/rand/pull/674
[#675]: https://github.com/rust-random/rand/pull/675
[#685]: https://github.com/rust-random/rand/pull/685
[#686]: https://github.com/rust-random/rand/pull/686
[#691]: https://github.com/rust-random/rand/pull/691
[#696]: https://github.com/rust-random/rand/pull/696
[#703]: https://github.com/rust-random/rand/pull/703
[#711]: https://github.com/rust-random/rand/pull/711

## [0.5.5] - 2018-08-07

### Documentation
- Fix links in documentation ([#582])

## [0.5.4] - 2018-07-11

### Platform support
- Make `OsRng` work via WASM/stdweb for WebWorkers

## [0.5.3] - 2018-06-26

### Platform support
- OpenBSD, Bitrig: fix compilation (broken in 0.5.1) ([#530])

## [0.5.2] - 2018-06-18

### Platform support
- Hide `OsRng` and `JitterRng` on unsupported platforms ([#512]; fixes [#503])

## [0.5.1] - 2018-06-08

### New distributions
- Added Cauchy distribution. ([#474], [#486])
- Added Pareto distribution. ([#495])

### Platform support and `OsRng`
- Remove blanket Unix implementation. ([#484])
- Remove Wasm unimplemented stub. ([#484])
- Dragonfly BSD: read from `/dev/random`. ([#484])
- Bitrig: use `getentropy` like OpenBSD. ([#484])
- Solaris: (untested) use `getrandom` if available, otherwise `/dev/random`. ([#484])
- Emscripten, `stdweb`: split the read up in chunks. ([#484])
- Emscripten, Haiku: don't do an extra blocking read from `/dev/random`. ([#484])
- Linux, NetBSD, Solaris: read in blocking mode on first use in `fill_bytes`. ([#484])
- Fuchsia, CloudABI: fix compilation (broken in Rand 0.5). ([#484])

## [0.5.0] - 2018-05-21

### Crate features and organisation
- Minimum Rust version update: 1.22.0. ([#239])
- Create a separate `rand_core` crate. ([#288])
- Deprecate `rand_derive`. ([#256])
- Add `prelude` (and module reorganisation). ([#435])
- Add `log` feature. Logging is now available in `JitterRng`, `OsRng`, `EntropyRng` and `ReseedingRng`. ([#246])
- Add `serde1` feature for some PRNGs. ([#189])
- `stdweb` feature for `OsRng` support on WASM via stdweb. ([#272], [#336])

### `Rng` trait
- Split `Rng` into `RngCore` and `Rng` extension trait. `next_u32`, `next_u64` and `fill_bytes` are now part of `RngCore`. ([#265])
- Add `Rng::sample`. ([#256])
- Deprecate `Rng::gen_weighted_bool`. ([#308])
- Add `Rng::gen_bool`. ([#308])
- Remove `Rng::next_f32` and `Rng::next_f64`. ([#273])
- Add optimized `Rng::fill` and `Rng::try_fill` methods. ([#247])
- Deprecate `Rng::gen_iter`. ([#286])
- Deprecate `Rng::gen_ascii_chars`. ([#279])

### `rand_core` crate
- `rand` now depends on new `rand_core` crate ([#288])
- `RngCore` and `SeedableRng` are now part of `rand_core`. ([#288])
- Add modules to help implementing RNGs `impl` and `le`. ([#209], [#228])
- Add `Error` and `ErrorKind`. ([#225])
- Add `CryptoRng` marker trait. ([#273])
- Add `BlockRngCore` trait. ([#281])
- Add `BlockRng` and `BlockRng64` wrappers to help implementations. ([#281], [#325])
- Revise the `SeedableRng` trait. ([#233])
- Remove default implementations for `RngCore::next_u64` and `RngCore::fill_bytes`. ([#288])
- Add `RngCore::try_fill_bytes`. ([#225])

### Other traits and types
- Add `FromEntropy` trait. ([#233], [#375])
- Add `SmallRng` wrapper. ([#296])
- Rewrite `ReseedingRng` to only work with `BlockRngCore` (substantial performance improvement). ([#281])
- Deprecate `weak_rng`. Use `SmallRng` instead. ([#296])
- Deprecate `AsciiGenerator`. ([#279])

### Random number generators
- Switch `StdRng` and `thread_rng` to HC-128. ([#277])
- `StdRng` must now be created with `from_entropy` instead of `new`
- Change `thread_rng` reseeding threshold to 32 MiB. ([#277])
- PRNGs no longer implement `Copy`. ([#209])
- `Debug` implementations no longer show internals. ([#209])
- Implement `Clone` for `ReseedingRng`, `JitterRng`, OsRng`. ([#383], [#384])
- Implement serialization for `XorShiftRng`, `IsaacRng` and `Isaac64Rng` under the `serde1` feature. ([#189])
- Implement `BlockRngCore` for `ChaChaCore` and `Hc128Core`. ([#281])
- All PRNGs are now portable across big- and little-endian architectures. ([#209])
- `Isaac64Rng::next_u32` no longer throws away half the results. ([#209])
- Add `IsaacRng::new_from_u64` and `Isaac64Rng::new_from_u64`. ([#209])
- Add the HC-128 CSPRNG `Hc128Rng`. ([#210])
- Change ChaCha20 to have 64-bit counter and 64-bit stream. ([#349])
- Changes to `JitterRng` to get its size down from 2112 to 24 bytes. ([#251])
- Various performance improvements to all PRNGs.

### Platform support and `OsRng`
- Add support for CloudABI. ([#224])
- Remove support for NaCl. ([#225])
- WASM support for `OsRng` via stdweb, behind the `stdweb` feature. ([#272], [#336])
- Use `getrandom` on more platforms for Linux, and on Android. ([#338])
- Use the `SecRandomCopyBytes` interface on macOS. ([#322])
- On systems that do not have a syscall interface, only keep a single file descriptor open for `OsRng`. ([#239])
- On Unix, first try a single read from `/dev/random`, then `/dev/urandom`. ([#338])
- Better error handling and reporting in `OsRng` (using new error type). ([#225])
- `OsRng` now uses non-blocking when available. ([#225])
- Add `EntropyRng`, which provides `OsRng`, but has `JitterRng` as a fallback. ([#235])

### Distributions
- New `Distribution` trait. ([#256])
- Add `Distribution::sample_iter` and `Rng::::sample_iter`. ([#361])
- Deprecate `Rand`, `Sample` and `IndependentSample` traits. ([#256])
- Add a `Standard` distribution (replaces most `Rand` implementations). ([#256])
- Add `Binomial` and `Poisson` distributions. ([#96])
- Add `Bernoulli` dsitribution. ([#411])
- Add `Alphanumeric` distribution. ([#279])
- Remove `Closed01` distribution, add `OpenClosed01`. ([#274], [#420])
- Rework `Range` type, making it possible to implement it for user types. ([#274])
- Rename `Range` to `Uniform`. ([#395])
- Add `Uniform::new_inclusive` for inclusive ranges. ([#274])
- Use widening multiply method for much faster integer range reduction. ([#274])
- `Standard` distribution for `char` uses `Uniform` internally. ([#274])
- `Standard` distribution for `bool` uses sign test. ([#274])
- Implement `Standard` distribution for `Wrapping<T>`. ([#436])
- Implement `Uniform` distribution for `Duration`. ([#427])

[#96]: https://github.com/rust-random/rand/pull/96
[#189]: https://github.com/rust-random/rand/pull/189
[#209]: https://github.com/rust-random/rand/pull/209
[#210]: https://github.com/rust-random/rand/pull/210
[#224]: https://github.com/rust-random/rand/pull/224
[#225]: https://github.com/rust-random/rand/pull/225
[#228]: https://github.com/rust-random/rand/pull/228
[#233]: https://github.com/rust-random/rand/pull/233
[#235]: https://github.com/rust-random/rand/pull/235
[#239]: https://github.com/rust-random/rand/pull/239
[#246]: https://github.com/rust-random/rand/pull/246
[#247]: https://github.com/rust-random/rand/pull/247
[#251]: https://github.com/rust-random/rand/pull/251
[#256]: https://github.com/rust-random/rand/pull/256
[#265]: https://github.com/rust-random/rand/pull/265
[#272]: https://github.com/rust-random/rand/pull/272
[#273]: https://github.com/rust-random/rand/pull/273
[#274]: https://github.com/rust-random/rand/pull/274
[#277]: https://github.com/rust-random/rand/pull/277
[#279]: https://github.com/rust-random/rand/pull/279
[#281]: https://github.com/rust-random/rand/pull/281
[#286]: https://github.com/rust-random/rand/pull/286
[#288]: https://github.com/rust-random/rand/pull/288
[#296]: https://github.com/rust-random/rand/pull/296
[#308]: https://github.com/rust-random/rand/pull/308
[#322]: https://github.com/rust-random/rand/pull/322
[#325]: https://github.com/rust-random/rand/pull/325
[#336]: https://github.com/rust-random/rand/pull/336
[#338]: https://github.com/rust-random/rand/pull/338
[#349]: https://github.com/rust-random/rand/pull/349
[#361]: https://github.com/rust-random/rand/pull/361
[#375]: https://github.com/rust-random/rand/pull/375
[#383]: https://github.com/rust-random/rand/pull/383
[#384]: https://github.com/rust-random/rand/pull/384
[#395]: https://github.com/rust-random/rand/pull/395
[#411]: https://github.com/rust-random/rand/pull/411
[#420]: https://github.com/rust-random/rand/pull/420
[#427]: https://github.com/rust-random/rand/pull/427
[#435]: https://github.com/rust-random/rand/pull/435
[#436]: https://github.com/rust-random/rand/pull/436
[#474]: https://github.com/rust-random/rand/pull/474
[#484]: https://github.com/rust-random/rand/pull/484
[#486]: https://github.com/rust-random/rand/pull/486
[#495]: https://github.com/rust-random/rand/pull/495
[#503]: https://github.com/rust-random/rand/pull/503
[#512]: https://github.com/rust-random/rand/pull/512
[#530]: https://github.com/rust-random/rand/pull/530
[#582]: https://github.com/rust-random/rand/pull/582
## [0.4.6] - 2019-01-26

### Platforms
- Fuchsia: Replaced fuchsia-zircon with fuchsia-cprng

## [0.4.5] - 2019-01-09

### Fixed
- Remove dependency on default features of `rand_core` ([#689])

## [0.4.4] - 2019-01-08

Version yanked due to semver-breaking change ([#688]).

### Added
- SGX support

## [0.4.3] - 2018-08-16

### Fixed
- Use correct syscall number for PowerPC ([#589])

## [0.4.2] - 2018-01-06

### Changed
- Use `winapi` on Windows
- Update for Fuchsia OS
- Remove dev-dependency on `log`

## [0.4.1] - 2017-12-17

### Added
- `no_std` support

## [0.4.0-pre.0] - 2017-12-11

### Added
- `JitterRng` added as a high-quality alternative entropy source using the system timer
- new `seq` module with `sample_iter`, `sample_slice`, etc.
- WASM support via dummy implementations (fail at run-time)
- Additional benchmarks, covering generators and new seq code

### Changed
- `thread_rng` uses `JitterRng` if seeding from system time fails (slower but more secure than previous method)

### Deprecated
- `sample` function deprecated (replaced by `sample_iter`)

[#589]: https://github.com/rust-random/rand/pull/589
[#688]: https://github.com/rust-random/rand/pull/688
[#689]: https://github.com/rust-random/rand/pull/689

## [0.3.22] - 2018-02-05
Code replaced with a compatibility layer over rand 0.4.

## [0.3.20] - 2018-01-06
### Changed
- Remove dev-dependency on `log`
- Update `fuchsia-zircon` dependency to 0.3.2


## [0.3.19] - 2017-12-27
### Changed
- Require `log <= 0.3.8` for dev builds
- Update `fuchsia-zircon` dependency to 0.3
- Fix broken links in docs (to unblock compiler docs testing CI)


## [0.3.18] - 2017-11-06
### Changed
- `thread_rng` is seeded from the system time if `OsRng` fails
- `weak_rng` now uses `thread_rng` internally


## [0.3.17] - 2017-10-07
### Changed
 - Fuchsia: Magenta was renamed Zircon

## [0.3.16] - 2017-07-27
### Added
- Implement Debug for mote non-public types
- implement `Rand` for (i|u)i128
- Support for Fuchsia

### Changed
- Add inline attribute to SampleRange::construct_range.
  This improves the benchmark for sample in 11% and for shuffle in 16%.
- Use `RtlGenRandom` instead of `CryptGenRandom`


## [0.3.15] - 2016-11-26
### Added
- Add `Rng` trait method `choose_mut`
- Redox support

### Changed
- Use `arc4rand` for `OsRng` on FreeBSD.
- Use `arc4random(3)` for `OsRng` on OpenBSD.

### Fixed
- Fix filling buffers 4 GiB or larger with `OsRng::fill_bytes` on Windows


## [0.3.14] - 2016-02-13
### Fixed
- Inline definitions from winapi/advapi32, which decreases build times


## [0.3.13] - 2016-01-09
### Fixed
- Compatible with Rust 1.7.0-nightly (needed some extra type annotations)


## [0.3.12] - 2015-11-09
### Changed
- Replaced the methods in `next_f32` and `next_f64` with the technique described
  Saito & Matsumoto at MCQMC'08. The new method should exhibit a slightly more
  uniform distribution.
- Depend on libc 0.2

### Fixed
- Fix iterator protocol issue in `rand::sample`


## [0.3.11] - 2015-08-31
### Added
- Implement `Rand` for arrays with n <= 32


## [0.3.10] - 2015-08-17
### Added
- Support for NaCl platforms

### Changed
- Allow `Rng` to be `?Sized`, impl for `&mut R` and `Box<R>` where `R: ?Sized + Rng`


## [0.3.9] - 2015-06-18
### Changed
- Use `winapi` for Windows API things

### Fixed
- Fixed test on stable/nightly
- Fix `getrandom` syscall number for aarch64-unknown-linux-gnu


## [0.3.8] - 2015-04-23
### Changed
- `log` is a dev dependency

### Fixed
- Fix race condition of atomics in `is_getrandom_available`


## [0.3.7] - 2015-04-03
### Fixed
- Derive Copy/Clone changes


## [0.3.6] - 2015-04-02
### Changed
- Move to stable Rust!


## [0.3.5] - 2015-04-01
### Fixed
- Compatible with Rust master


## [0.3.4] - 2015-03-31
### Added
- Implement Clone for `Weighted`

### Fixed
- Compatible with Rust master


## [0.3.3] - 2015-03-26
### Fixed
- Fix compile on Windows


## [0.3.2] - 2015-03-26


## [0.3.1] - 2015-03-26
### Fixed
- Fix compile on Windows


## [0.3.0] - 2015-03-25
### Changed
- Update to use log version 0.3.x


## [0.2.1] - 2015-03-22
### Fixed
- Compatible with Rust master
- Fixed iOS compilation


## [0.2.0] - 2015-03-06
### Fixed
- Compatible with Rust master (move from `old_io` to `std::io`)


## [0.1.4] - 2015-03-04
### Fixed
- Compatible with Rust master (use wrapping ops)


## [0.1.3] - 2015-02-20
### Fixed
- Compatible with Rust master

### Removed
- Removed Copy implementations from RNGs


## [0.1.2] - 2015-02-03
### Added
- Imported functionality from `std::rand`, including:
  - `StdRng`, `SeedableRng`, `TreadRng`, `weak_rng()`
  - `ReaderRng`: A wrapper around any Reader to treat it as an RNG.
- Imported documentation from `std::rand`
- Imported tests from `std::rand`


## 0.1.1 - 2015-02-03
### Added
- Migrate to a cargo-compatible directory structure.

### Fixed
- Do not use entropy during `gen_weighted_bool(1)`


## Rust 0.12.0 - 2014-10-09
### Added
- Impl Rand for tuples of arity 11 and 12
- Include ChaCha pseudorandom generator
- Add `next_f64` and `next_f32` to Rng
- Implement Clone for PRNGs

### Changed
- Rename `TaskRng` to `ThreadRng` and `task_rng` to `thread_rng` (since a
  runtime is removed from Rust).

### Fixed
- Improved performance of ISAAC and ISAAC64 by 30% and 12 % respectively, by
  informing the optimiser that indexing is never out-of-bounds.

### Removed
- Removed the Deprecated `choose_option`


## Rust 0.11.0 - 2014-07-02
### Added
- document when to use `OSRng` in cryptographic context, and explain why we use `/dev/urandom` instead of `/dev/random`
- `Rng::gen_iter()` which will return an infinite stream of random values
- `Rng::gen_ascii_chars()` which will return an infinite stream of random ascii characters

### Changed
- Now only depends on libcore!
- Remove `Rng.choose()`, rename `Rng.choose_option()` to `.choose()`
- Rename OSRng to OsRng
- The WeightedChoice structure is no longer built with a `Vec<Weighted<T>>`,
  but rather a `&mut [Weighted<T>]`. This means that the WeightedChoice
  structure now has a lifetime associated with it.
- The `sample` method on `Rng` has been moved to a top-level function in the
  `rand` module due to its dependence on `Vec`.

### Removed
- `Rng::gen_vec()` was removed. Previous behavior can be regained with
  `rng.gen_iter().take(n).collect()`
- `Rng::gen_ascii_str()` was removed. Previous behavior can be regained with
  `rng.gen_ascii_chars().take(n).collect()`
- {IsaacRng, Isaac64Rng, XorShiftRng}::new() have all been removed. These all
  relied on being able to use an OSRng for seeding, but this is no longer
  available in librand (where these types are defined). To retain the same
  functionality, these types now implement the `Rand` trait so they can be
  generated with a random seed from another random number generator. This allows
  the stdlib to use an OSRng to create seeded instances of these RNGs.
- Rand implementations for `Box<T>` and `@T` were removed. These seemed to be
  pretty rare in the codebase, and it allows for librand to not depend on
  liballoc.  Additionally, other pointer types like Rc<T> and Arc<T> were not
  supported.
- Remove a slew of old deprecated functions


## [Rust 0.10] - 2014-04-03
### Changed
- replace `Rng.shuffle's` functionality with `.shuffle_mut`
- bubble up IO errors when creating an OSRng

### Fixed
- Use `fill()` instead of `read()`
- Rewrite OsRng in Rust for windows

## 0.10-pre - 2014-03-02
### Added
- Separate `rand` out of the standard library

[Unreleased]: https://github.com/rust-random/rand/compare/0.10.1...HEAD
[0.10.1]: https://github.com/rust-random/rand/compare/0.10.0...0.10.1
[0.10.0]: https://github.com/rust-random/rand/compare/0.9.2...0.10.0
[0.9.2]: https://github.com/rust-random/rand/compare/0.9.1...0.9.2
[0.9.1]: https://github.com/rust-random/rand/compare/0.9.0...0.9.1
[0.9.0]: https://github.com/rust-random/rand/compare/0.8.5...0.9.0
[0.8.5]: https://github.com/rust-random/rand/compare/0.8.4...0.8.5
[0.8.4]: https://github.com/rust-random/rand/compare/0.8.3...0.8.4
[0.8.3]: https://github.com/rust-random/rand/compare/0.8.2...0.8.3
[0.8.2]: https://github.com/rust-random/rand/compare/0.8.1...0.8.2
[0.8.1]: https://github.com/rust-random/rand/compare/0.8.0...0.8.1
[0.8.0]: https://github.com/rust-random/rand/compare/0.7.3...0.8.0
[0.7.3]: https://github.com/rust-random/rand/compare/0.7.2...0.7.3
[0.7.2]: https://github.com/rust-random/rand/compare/0.7.1...0.7.2
[0.7.1]: https://github.com/rust-random/rand/compare/0.7.0...0.7.1
[0.7.0]: https://github.com/rust-random/rand/compare/0.6.5...0.7.0
[0.6.5]: https://github.com/rust-random/rand/compare/0.6.4...0.6.5
[0.6.4]: https://github.com/rust-random/rand/compare/0.6.3...0.6.4
[0.6.3]: https://github.com/rust-random/rand/compare/0.6.2...0.6.3
[0.6.2]: https://github.com/rust-random/rand/compare/0.6.1...0.6.2
[0.6.1]: https://github.com/rust-random/rand/compare/0.6.0...0.6.1
[0.6.0]: https://github.com/rust-random/rand/compare/0.5.5...0.6.0
[0.5.5]: https://github.com/rust-random/rand/compare/0.5.4...0.5.5
[0.5.4]: https://github.com/rust-random/rand/compare/0.5.3...0.5.4
[0.5.3]: https://github.com/rust-random/rand/compare/0.5.2...0.5.3
[0.5.2]: https://github.com/rust-random/rand/compare/0.5.1...0.5.2
[0.5.1]: https://github.com/rust-random/rand/compare/0.5.0...0.5.1
[0.5.0]: https://github.com/rust-random/rand/compare/0.4.6...0.5.0
[0.4.6]: https://github.com/rust-random/rand/compare/0.4.5...0.4.6
[0.4.5]: https://github.com/rust-random/rand/compare/0.4.4...0.4.5
[0.4.4]: https://github.com/rust-random/rand/compare/0.4.3...0.4.4
[0.4.3]: https://github.com/rust-random/rand/compare/0.4.2...0.4.3
[0.4.2]: https://github.com/rust-random/rand/compare/0.4.1...0.4.2
[0.4.1]: https://github.com/rust-random/rand/compare/0.4.0-pre.0...0.4.1
[0.4.0-pre.0]: https://github.com/rust-random/rand/compare/0.3.22...0.4.0-pre.0
[0.3.22]: https://github.com/rust-random/rand/compare/0.3.20...0.3.22
[0.3.20]: https://github.com/rust-random/rand/compare/0.3.19...0.3.20
[0.3.19]: https://github.com/rust-random/rand/compare/0.3.18...0.3.19
[0.3.18]: https://github.com/rust-random/rand/compare/0.3.17...0.3.18
[0.3.17]: https://github.com/rust-random/rand/compare/0.3.16...0.3.17
[0.3.16]: https://github.com/rust-random/rand/compare/0.3.15...0.3.16
[0.3.15]: https://github.com/rust-random/rand/compare/0.3.14...0.3.15
[0.3.14]: https://github.com/rust-random/rand/compare/0.3.13...0.3.14
[0.3.13]: https://github.com/rust-random/rand/compare/0.3.12...0.3.13
[0.3.12]: https://github.com/rust-random/rand/compare/0.3.11...0.3.12
[0.3.11]: https://github.com/rust-random/rand/compare/0.3.10...0.3.11
[0.3.10]: https://github.com/rust-random/rand/compare/0.3.9...0.3.10
[0.3.9]: https://github.com/rust-random/rand/compare/0.3.8...0.3.9
[0.3.8]: https://github.com/rust-random/rand/compare/0.3.7...0.3.8
[0.3.7]: https://github.com/rust-random/rand/compare/0.3.6...0.3.7
[0.3.6]: https://github.com/rust-random/rand/compare/0.3.5...0.3.6
[0.3.5]: https://github.com/rust-random/rand/compare/0.3.4...0.3.5
[0.3.4]: https://github.com/rust-random/rand/compare/0.3.3...0.3.4
[0.3.3]: https://github.com/rust-random/rand/compare/0.3.2...0.3.3
[0.3.2]: https://github.com/rust-random/rand/compare/0.3.1...0.3.2
[0.3.1]: https://github.com/rust-random/rand/compare/0.3.0...0.3.1
[0.3.0]: https://github.com/rust-random/rand/compare/0.2.1...0.3.0
[0.2.1]: https://github.com/rust-random/rand/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/rust-random/rand/compare/0.1.4...0.2.0
[0.1.3]: https://github.com/rust-random/rand/compare/0.1.3...0.1.4
[0.1.3]: https://github.com/rust-random/rand/compare/0.1.2...0.1.3
[0.1.2]: https://github.com/rust-random/rand/compare/0.1.1...0.1.2
