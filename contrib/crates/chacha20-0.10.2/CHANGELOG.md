# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.10.2 (2026-08-27)
### Fixed
- Use of SSE4.1 intrinsic in SSE2 backend of RNG and legacy (64-bit counter) variants ([#580])

[#580]: https://github.com/RustCrypto/stream-ciphers/pull/580

## 0.10.1 (2026-06-24)
### Added
- `ChaCha20LegacyCore` type and `Nonce` type alias ([#570])

[#570]: https://github.com/RustCrypto/stream-ciphers/pull/570

## 0.10.0 (2026-02-07)
### Added
- `rand_core` v0.10 support ([#333], [#513])
- 64-bit counter support ([#439])
- `{get,set}_block_pos` inherent methods ([#516])
- AVX-512 backend - requires `--cfg chacha20_avx512` to enable ([#477])
- Inherent methods for RNG state (de)serialization ([#541])

### Changed
- Bump `cipher` from `0.4` to `0.5` - replaces `generic-array` with `hybrid-array` ([#338], [#521])
- Edition changed to 2024 and MSRV bumped to 1.85 ([#397])
- Relax MSRV policy and allow MSRV bumps in patch releases
- Bump `cpufeatures` to v0.3 ([#530])
- `--cfg chacha20_backend="..."` replaces previous `chacha20_force*` ([#520])
  - `chacha20_force_avx2` => `chacha20_backend="avx2"`
  - `chacha20_force_avx512` => `chacha20_backend="avx512"`
  - `chacha20_force_soft` => `chacha20_backend="soft"`
  - `chacha20_force_sse2` => `chacha20_backend="sse2"`

### Removed
- `chacha20_force_neon` cfg attribute - now on-by-default for supported targets ([#361])
- `std` feature ([#397])
- `Clone` impls ([#462])

[#333]: https://github.com/RustCrypto/stream-ciphers/pull/333
[#338]: https://github.com/RustCrypto/stream-ciphers/pull/338
[#361]: https://github.com/RustCrypto/stream-ciphers/pull/361
[#397]: https://github.com/RustCrypto/stream-ciphers/pull/397
[#439]: https://github.com/RustCrypto/stream-ciphers/pull/439
[#462]: https://github.com/RustCrypto/stream-ciphers/pull/462
[#477]: https://github.com/RustCrypto/stream-ciphers/pull/477
[#513]: https://github.com/RustCrypto/stream-ciphers/pull/513
[#516]: https://github.com/RustCrypto/stream-ciphers/pull/516
[#520]: https://github.com/RustCrypto/stream-ciphers/pull/520
[#521]: https://github.com/RustCrypto/stream-ciphers/pull/521
[#530]: https://github.com/RustCrypto/stream-ciphers/pull/530
[#541]: https://github.com/RustCrypto/stream-ciphers/pull/541

## 0.9.1 (2023-04-01)
### Added
- NEON support via `chacha20_force_neon` cfg attribute ([#310], [#317])

[#310]: https://github.com/RustCrypto/stream-ciphers/pull/310
[#317]: https://github.com/RustCrypto/stream-ciphers/pull/317

## 0.9.0 (2022-02-21)
### Added
- `chacha20_force_soft`, `chacha20_force_sse2`, and `chacha20_force_avx2`
configuration flags ([#293])

### Changed
- Bump `cipher` dependency to v0.4 ([#276])

### Fixed
- Minimal versions build ([#290])

### Removed
- `neon`, `force-soft`, `expose-core`, `hchacha`, `legacy`, and `rng` features ([#276], [#293])

[#276]: https://github.com/RustCrypto/stream-ciphers/pull/276
[#290]: https://github.com/RustCrypto/stream-ciphers/pull/290
[#293]: https://github.com/RustCrypto/stream-ciphers/pull/293

## 0.8.2 (2022-07-07)
### Changed
- Unpin `zeroize` dependency ([#301])

[#301]: https://github.com/RustCrypto/stream-ciphers/pull/301

## 0.8.1 (2021-08-30)
### Added
- NEON implementation for aarch64 ([#274])

[#274]: https://github.com/RustCrypto/stream-ciphers/pull/274

## 0.8.0 (2021-08-29)
### Added
- SSE2 autodetection support ([#270])

### Changed
- AVX2 performance improvements ([#267], [#267])
- MSRV 1.51+ ([#267])
- Lock to `zeroize` <1.5 ([#269])

### Removed
- `xchacha` feature: all `XChaCha*` types are now available by-default ([#271])

[#267]: https://github.com/RustCrypto/stream-ciphers/pull/267
[#269]: https://github.com/RustCrypto/stream-ciphers/pull/269
[#270]: https://github.com/RustCrypto/stream-ciphers/pull/270
[#271]: https://github.com/RustCrypto/stream-ciphers/pull/271

## 0.7.3 (2021-08-27)
### Changed
- Improve AVX2 performance ([#261])
- Bump `cpufeatures` to v0.2 ([#265])

[#261]: https://github.com/RustCrypto/stream-ciphers/pull/261
[#265]: https://github.com/RustCrypto/stream-ciphers/pull/265

## 0.7.2 (2021-07-20)
### Changed
- Pin `zeroize` dependency to v1.3 ([#256])

[#256]: https://github.com/RustCrypto/stream-ciphers/pull/256

## 0.7.1 (2021-04-29)
### Added
- `hchacha` feature ([#234])

[#234]: https://github.com/RustCrypto/stream-ciphers/pull/234

## 0.7.0 (2021-04-29) [YANKED]
### Added
- AVX2 detection; MSRV 1.49+ ([#200], [#212])
- `XChaCha8` and `XChaCha12` ([#215])

### Changed
- Full 64-bit counters ([#217])
- Bump `cipher` crate dependency to v0.3 release ([#226])

### Fixed
- `rng` feature on big endian platforms ([#202])
- Stream-length overflow check ([#216])

### Removed
- `Clone` impls on RNGs ([#220])

[#200]: https://github.com/RustCrypto/stream-ciphers/pull/200
[#202]: https://github.com/RustCrypto/stream-ciphers/pull/202
[#212]: https://github.com/RustCrypto/stream-ciphers/pull/212
[#215]: https://github.com/RustCrypto/stream-ciphers/pull/215
[#216]: https://github.com/RustCrypto/stream-ciphers/pull/216
[#217]: https://github.com/RustCrypto/stream-ciphers/pull/217
[#220]: https://github.com/RustCrypto/stream-ciphers/pull/220
[#226]: https://github.com/RustCrypto/stream-ciphers/pull/226

## 0.6.0 (2020-10-16)
### Changed
- Rename `Cipher` to `ChaCha` ([#177])
- Replace `block-cipher`/`stream-cipher` with `cipher` crate ([#177])

[#177]: https://github.com/RustCrypto/stream-ciphers/pull/177

## 0.5.0 (2020-08-25)
### Changed
- Bump `stream-cipher` dependency to v0.7 ([#161], [#164])

[#161]: https://github.com/RustCrypto/stream-ciphers/pull/161
[#164]: https://github.com/RustCrypto/stream-ciphers/pull/164

## 0.4.3 (2020-06-11)
### Changed
- Documentation improvements ([#153], [#154], [#155])

[#153]: https://github.com/RustCrypto/stream-ciphers/pull/155
[#154]: https://github.com/RustCrypto/stream-ciphers/pull/155
[#155]: https://github.com/RustCrypto/stream-ciphers/pull/155

## 0.4.2 (2020-06-11)
### Added
- Documentation improvements ([#149])
- `Key`, `Nonce`, `XNonce`, and `LegacyNonce` type aliases ([#147])

[#149]: https://github.com/RustCrypto/stream-ciphers/pull/149
[#147]: https://github.com/RustCrypto/stream-ciphers/pull/147

## 0.4.1 (2020-06-06)
### Fixed
- Links in documentation ([#142])

[#142]: https://github.com/RustCrypto/stream-ciphers/pull/142

## 0.4.0 (2020-06-06)
### Changed
- Upgrade to the `stream-cipher` v0.4 crate ([#121], [#138])

[#138]: https://github.com/RustCrypto/stream-ciphers/pull/138
[#121]: https://github.com/RustCrypto/stream-ciphers/pull/121

## 0.3.4 (2020-03-02)
### Fixed
- Avoid accidental `alloc` and `std` linking ([#105])

[#105]: https://github.com/RustCrypto/stream-ciphers/pull/105

## 0.3.3 (2020-01-18)
### Changed
- Replace macros with `Rounds` trait + generics ([#100])

### Fixed
- Fix warnings when building with `rng` feature alone ([#99])

[#99]: https://github.com/RustCrypto/stream-ciphers/pull/99
[#100]: https://github.com/RustCrypto/stream-ciphers/pull/100

## 0.3.2 (2020-01-17)
### Added
- `CryptoRng` marker on all `ChaCha*Rng` types ([#91])

[#91]: https://github.com/RustCrypto/stream-ciphers/pull/91

## 0.3.1 (2020-01-16)
### Added
- Parallelize AVX2 backend ([#87])
- Benchmark for `ChaCha20Rng` ([#87])

### Fixed
- Fix broken buffering logic ([#86])

[#86]: https://github.com/RustCrypto/stream-ciphers/pull/86
[#87]: https://github.com/RustCrypto/stream-ciphers/pull/87

## 0.3.0 (2020-01-15) [YANKED]

NOTE: This release was yanked due to a showstopper bug in the newly added
buffering logic which when seeking in the keystream could result in plaintexts
being clobbered with the keystream instead of XOR'd correctly.

The bug was addressed in v0.3.1 ([#86]).

### Added
- AVX2 accelerated implementation ([#83])
- ChaCha8 and ChaCha20 reduced round variants ([#84])

### Changed
- Simplify portable implementation ([#76])
- Make 2018 edition crate; MSRV 1.34+ ([#77])
- Replace `salsa20-core` dependency with `ctr`-derived buffering ([#81])

### Removed
- `byteorder` dependency ([#80])

[#76]: https://github.com/RustCrypto/stream-ciphers/pull/76
[#77]: https://github.com/RustCrypto/stream-ciphers/pull/77
[#80]: https://github.com/RustCrypto/stream-ciphers/pull/80
[#81]: https://github.com/RustCrypto/stream-ciphers/pull/81
[#83]: https://github.com/RustCrypto/stream-ciphers/pull/83
[#84]: https://github.com/RustCrypto/stream-ciphers/pull/84

## 0.2.3 (2019-10-23)
### Security
- Ensure block counter < MAX_BLOCKS ([#68])

[#68]: https://github.com/RustCrypto/stream-ciphers/pull/68

## 0.2.2 (2019-10-22)
### Added
- SSE2 accelerated implementation ([#61])

[#61]: https://github.com/RustCrypto/stream-ciphers/pull/61

## 0.2.1 (2019-08-19)
### Added
- Add `MAX_BLOCKS` and `BLOCK_SIZE` constants ([#47])

[#47]: https://github.com/RustCrypto/stream-ciphers/pull/47

## 0.2.0 (2019-08-18)
### Added
- `impl SyncStreamCipher` ([#39])
- `XChaCha20` ([#36])
- Support for 12-byte nonces ala RFC 8439 ([#19])

### Changed
- Refactor around a `ctr`-like type ([#44])
- Extract and encapsulate `Cipher` type ([#43])
- Switch tests to use `new_sync_test!` ([#42])
- Refactor into `ChaCha20` and `ChaCha20Legacy` ([#25])

### Fixed
- Fix `zeroize` cargo feature ([#21])
- Fix broken Cargo feature attributes ([#21])

[#44]: https://github.com/RustCrypto/stream-ciphers/pull/44
[#43]: https://github.com/RustCrypto/stream-ciphers/pull/43
[#42]: https://github.com/RustCrypto/stream-ciphers/pull/42
[#39]: https://github.com/RustCrypto/stream-ciphers/pull/39
[#36]: https://github.com/RustCrypto/stream-ciphers/pull/36
[#25]: https://github.com/RustCrypto/stream-ciphers/pull/25
[#21]: https://github.com/RustCrypto/stream-ciphers/pull/21
[#19]: https://github.com/RustCrypto/stream-ciphers/pull/19

## 0.1.0 (2019-06-24)

- Initial release
