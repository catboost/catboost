# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.51](https://github.com/Nullus157/async-compression/compare/async-compression-v0.4.50...async-compression-v0.4.51) - 2026-10-17

### Added

- *(zstd)* add `mzstd` v0.2.5 ([#505](https://github.com/Nullus157/async-compression/pull/505))

### Other

- add keywords to compression crates ([#502](https://github.com/Nullus157/async-compression/pull/502))

## [0.4.50](https://github.com/Nullus157/async-compression/compare/async-compression-v0.4.49...async-compression-v0.4.50) - 2026-09-30

### Other

- *(deps)* update brotli requirement from 8 to 9 ([#499](https://github.com/Nullus157/async-compression/pull/499))

## [0.4.49](https://github.com/Nullus157/async-compression/compare/async-compression-v0.4.48...async-compression-v0.4.49) - 2026-09-29

### Added

- *(brotli)* add optional mbrotli backend ([#492](https://github.com/Nullus157/async-compression/pull/492))

## [0.4.48](https://github.com/Nullus157/async-compression/compare/async-compression-v0.4.47...async-compression-v0.4.48) - 2026-09-16

### Fixed

- *(gzip)* reject invalid header before the full 10 bytes arrive ([#489](https://github.com/Nullus157/async-compression/pull/489))

## [0.4.47](https://github.com/Nullus157/async-compression/compare/async-compression-v0.4.46...async-compression-v0.4.47) - 2026-09-14

### Fixed

- *(zstd)* avoid no-progress errors while awaiting input ([#486](https://github.com/Nullus157/async-compression/pull/486))

## [0.4.46](https://github.com/Nullus157/async-compression/compare/async-compression-v0.4.45...async-compression-v0.4.46) - 2026-09-09

### Other

- Return UnexpectedEof when deflate64 does not progress ([#483](https://github.com/Nullus157/async-compression/pull/483))

## [0.4.45](https://github.com/Nullus157/async-compression/compare/async-compression-v0.4.44...async-compression-v0.4.45) - 2026-09-07

### Other

- *(deps)* update zstd requirement from 0.13 to 0.14 ([#481](https://github.com/Nullus157/async-compression/pull/481))

## [0.4.44](https://github.com/Nullus157/async-compression/compare/async-compression-v0.4.43...async-compression-v0.4.44) - 2026-09-02

### Other

- updated the following local packages: compression-core, compression-codecs

## [0.4.43](https://github.com/Nullus157/async-compression/compare/async-compression-v0.4.42...async-compression-v0.4.43) - 2026-07-28

### Other

- Fix hang when decoding a corrupt subsequent zstd frame ([#470](https://github.com/Nullus157/async-compression/pull/470))

## [0.4.42](https://github.com/Nullus157/async-compression/compare/async-compression-v0.4.41...async-compression-v0.4.42) - 2026-04-21

### Other

- Include license files in published crates ([#467](https://github.com/Nullus157/async-compression/pull/467))
- Fix async-compression CHANGELOG ordering ([#463](https://github.com/Nullus157/async-compression/pull/463))

## [0.4.41](https://github.com/Nullus157/async-compression/compare/async-compression-v0.4.40...async-compression-v0.4.41) - 2026-02-25

### Fixed

- On error the encoder/decoder now emit everything it was able to decode. ([#454](https://github.com/Nullus157/async-compression/pull/454))

## [0.4.40](https://github.com/Nullus157/async-compression/compare/async-compression-v0.4.39...async-compression-v0.4.40) - 2026-02-18

### Other

- Fix update of bytes read in the encoder state machine. ([#456](https://github.com/Nullus157/async-compression/pull/456))
- *(deps)* update rand requirement from 0.9 to 0.10 ([#449](https://github.com/Nullus157/async-compression/pull/449))

## [0.4.39](https://github.com/Nullus157/async-compression/compare/async-compression-v0.4.38...async-compression-v0.4.39) - 2026-02-07

### Other

- Revert "Fix decoding of empty response streams." ([#446](https://github.com/Nullus157/async-compression/pull/446))

## [0.4.38](https://github.com/Nullus157/async-compression/compare/async-compression-v0.4.37...async-compression-v0.4.38) - 2026-02-06

### Other

- Fix decoding of empty response streams. ([#444](https://github.com/Nullus157/async-compression/pull/444))
- *(deps)* update proptest-derive requirement from 0.7 to 0.8 ([#442](https://github.com/Nullus157/async-compression/pull/442))

## [0.4.37](https://github.com/Nullus157/async-compression/compare/async-compression-v0.4.36...async-compression-v0.4.37) - 2026-01-10

### Other

- Set async-compression msrv ([#440](https://github.com/Nullus157/async-compression/pull/440))
- Fix BufWriter::flush_buf: propagate error before reset ([#438](https://github.com/Nullus157/async-compression/pull/438))
- Switch futures_core::ready to standard library api ([#436](https://github.com/Nullus157/async-compression/pull/436))

## [0.4.36](https://github.com/Nullus157/async-compression/compare/async-compression-v0.4.35...async-compression-v0.4.36) - 2025-12-11

### Other

- updated the following local packages: compression-codecs

## [0.4.35](https://github.com/Nullus157/async-compression/compare/async-compression-v0.4.34...async-compression-v0.4.35) - 2025-12-07

### Other

- updated the following local packages: compression-codecs

## [0.4.34](https://github.com/Nullus157/async-compression/compare/async-compression-v0.4.33...async-compression-v0.4.34) - 2025-11-21

### Added

- Add uninitialized output buffer support ([#414](https://github.com/Nullus157/async-compression/pull/414))

### Other

- Simplify `AsyncBufWrite` ([#423](https://github.com/Nullus157/async-compression/pull/423))
- Refactor `bufread::decoder`: Extract poll ready check ([#418](https://github.com/Nullus157/async-compression/pull/418))
- Optimize condition for removing written data in BufWriter ([#415](https://github.com/Nullus157/async-compression/pull/415))

## [0.4.33](https://github.com/Nullus157/async-compression/compare/async-compression-v0.4.32...async-compression-v0.4.33) - 2025-11-06

### Fixed

- `UnexpectedEof` on truncated input ([#412](https://github.com/Nullus157/async-compression/pull/412))

### Other

- Update async-compression to use codecs v2 ([#410](https://github.com/Nullus157/async-compression/pull/410))
- Dedup `write::Encoder` and simplify `write::Decoder` impl ([#409](https://github.com/Nullus157/async-compression/pull/409))
- Dedup `write::Decoder::poll_write` ([#408](https://github.com/Nullus157/async-compression/pull/408))
- Optimize `BufWriter::poll_*` method ([#407](https://github.com/Nullus157/async-compression/pull/407))
- Dedup `write::{AsyncBufWriter, BufWriter, Decoder}` ([#406](https://github.com/Nullus157/async-compression/pull/406))
- Early return in `do_poll_read` instead of keep polling ([#404](https://github.com/Nullus157/async-compression/pull/404))
- Deduplicate `bufread::Encoder` impl ([#402](https://github.com/Nullus157/async-compression/pull/402))
- Rename `impl_do_poll_read!` to `impl_decoder!` ([#403](https://github.com/Nullus157/async-compression/pull/403))
- *(deps)* update proptest-derive requirement from 0.6 to 0.7 ([#399](https://github.com/Nullus157/async-compression/pull/399))
- Deduplicate `generic::bufread::Decoder` impl of tokio/futures-io ([#391](https://github.com/Nullus157/async-compression/pull/391))
- Disable nightly feature `doc_auto_cfg` on docsrs ([#392](https://github.com/Nullus157/async-compression/pull/392))

## [0.4.32](https://github.com/Nullus157/async-compression/compare/async-compression-v0.4.31...async-compression-v0.4.32) - 2025-09-25

### Other

- updated the following local packages: compression-codecs

## [0.4.31](https://github.com/Nullus157/async-compression/compare/async-compression-v0.4.30...async-compression-v0.4.31) - 2025-09-25

### Other

- Flush compressed data out of encoders more often ([#383](https://github.com/Nullus157/async-compression/pull/383))

## [0.4.30](https://github.com/Nullus157/async-compression/compare/async-compression-v0.4.29...async-compression-v0.4.30) - 2025-08-31

### Other

- rm unused dep from async-compression and compression-codecs ([#381](https://github.com/Nullus157/async-compression/pull/381))

## [0.4.29](https://github.com/Nullus157/async-compression/compare/async-compression-v0.4.28...async-compression-v0.4.29) - 2025-08-28

### Other

- Update Deps.rs badge ([#380](https://github.com/Nullus157/async-compression/pull/380))
- move async-compression to crates/ ([#379](https://github.com/Nullus157/async-compression/pull/379))

## [0.4.28](https://github.com/Nullus157/async-compression/compare/async-compression-v0.4.27...async-compression-v0.4.28) - 2025-08-23

### Fixed

- fix wasi ci testing and update doc in README ([#367](https://github.com/Nullus157/async-compression/pull/367))

### Other

- Fix Cargo.toml: add back version for async-compression ([#372](https://github.com/Nullus157/async-compression/pull/372))
- Have separate package.version field for compression-* ([#369](https://github.com/Nullus157/async-compression/pull/369))
- Re-export compression_codecs as codecs ([#368](https://github.com/Nullus157/async-compression/pull/368))
- Fix breaking API change ([#366](https://github.com/Nullus157/async-compression/pull/366))
- Fix docs.rs build for compression-codecs ([#365](https://github.com/Nullus157/async-compression/pull/365))
- Separate codecs as a separate crate, allow direct configuration ([#363](https://github.com/Nullus157/async-compression/pull/363))
- *(deps)* bump actions/checkout from 4 to 5 ([#360](https://github.com/Nullus157/async-compression/pull/360))
- Fix doc link for futures-io ([#361](https://github.com/Nullus157/async-compression/pull/361))

## [0.4.27](https://github.com/Nullus157/async-compression/compare/v0.4.26...v0.4.27) - 2025-07-13

### Fixed

- Zstd fastest level now does compression. ([#357](https://github.com/Nullus157/async-compression/pull/357))

## [0.4.26](https://github.com/Nullus157/async-compression/compare/v0.4.25...v0.4.26) - 2025-07-13

### Features

- Add multi-thread support for XZ. ([#353](https://github.com/Nullus157/async-compression/pull/353))

## [0.4.25](https://github.com/Nullus157/async-compression/compare/v0.4.24...v0.4.25) - 2025-06-17

### Changed

- Update `bzip2` dependency to `0.6`. ([#350](https://github.com/Nullus157/async-compression/pull/350))

## [0.4.24](https://github.com/Nullus157/async-compression/compare/v0.4.23...v0.4.24) - 2025-06-09

### Other

- Fix clippy error
- narrow tokio requirement to account for RUSTSEC-2025-0023
- move constructors back to top of rustdoc pages

## [0.4.23](https://github.com/Nullus157/async-compression/compare/v0.4.22...v0.4.23) - 2025-04-21

### Changed

- Update `brotli` dependency to `8.0`.
- Update `liblzma` dependency to `0.4`.

## [0.4.22](https://github.com/Nullus157/async-compression/compare/v0.4.21...v0.4.22) - 2025-03-25

### Added

- Add LZ4 encoders and decoders.
- Expose `DeflateEncoder::{total_in, total_out}()` methods.


## [0.4.21](https://github.com/Nullus157/async-compression/compare/v0.4.20...v0.4.21) - 2025-03-15

### Fixed

- When flate encoding, do not mark internal state as flushed if it ran out of buffer space.
- Add debug assertion in `produce` method to check buffer capacity in implementations for `BufWriter`.

## [0.4.20](https://github.com/Nullus157/async-compression/compare/v0.4.19...v0.4.20) - 2025-02-28

### Added

- Add support for `wasm32-wasip1-*` targets.

## [0.4.19](https://github.com/Nullus157/async-compression/compare/v0.4.18...v0.4.19) - 2025-02-27

### Changed

- Update `bzip2` dependency to `0.5`.

### Fixed

- Ensure that flush finishes before continuing.

## [0.4.18](https://github.com/Nullus157/async-compression/compare/v0.4.17...v0.4.18) - 2024-11-23

### Fixed

- Adjust `Level::Precise` clamp range for flate2.

## [0.4.17](https://github.com/Nullus157/async-compression/compare/v0.4.16...v0.4.17) - 2024-10-20

### Fixed

- Fix occasional panics when consuming from pending buffers.

## [0.4.16](https://github.com/Nullus157/async-compression/compare/v0.4.15...v0.4.16) - 2024-10-16

### Other

- Implement pass-through `AsyncBufRead` on write-based encoders & decoders.

## [0.4.15](https://github.com/Nullus157/async-compression/compare/v0.4.14...v0.4.15) - 2024-10-13

### Feature
- Implement pass-through `AsyncRead` or `AsyncWrite` where appropriate.
- Relax `AsyncRead`/`AsyncWrite` bounds on `*::{get_ref, get_mut, get_pin_mut, into_inner}()` methods.

## [0.4.14](https://github.com/Nullus157/async-compression/compare/v0.4.13...v0.4.14) - 2024-10-10

### Fixed
- In Tokio-based decoders, attempt to decode from internal state even if nothing was read.

## [0.4.13](https://github.com/Nullus157/async-compression/compare/v0.4.12...v0.4.13) - 2024-10-02

### Feature
- Update `brotli` dependency to to `7`.

## [0.4.12](https://github.com/Nullus157/async-compression/compare/v0.4.11...v0.4.12) - 2024-07-21

### Feature
- Enable customizing Zstd decoding parameters.

## [0.4.11](https://github.com/Nullus157/async-compression/compare/v0.4.10...v0.4.11) - 2024-05-30

### Other
- Expose total_in/total_out from underlying flate2 encoder types.

## [0.4.10](https://github.com/Nullus157/async-compression/compare/v0.4.9...v0.4.10) - 2024-05-09

### Other
- *(deps)* update brotli requirement from 5.0 to 6.0 ([#274](https://github.com/Nullus157/async-compression/pull/274))
- Fix pipeline doc: Warn on unexpected cfgs instead of error ([#276](https://github.com/Nullus157/async-compression/pull/276))
- Update name of release-pr.yml
- Create release.yml
- Create release-pr.yml

## 0.4.9

 - bump dep brotli from 4.0 to 5.0

## 0.4.8

 - bump dep brotli from 3.3 to 4.0

## 0.4.7

- Flush available data in decoder even when there's no incoming input.

## 0.4.6

- Return errors instead of panicking in all encode and decode operations.

## 0.4.5

- Add `{Lzma, Xz}Decoder::with_mem_limit()` methods.

## 0.4.4

- Update `zstd` dependency to `0.13`.

## 0.4.3

- Implement `Default` for `brotli::EncoderParams`.

## 0.4.2

- Add top-level `brotli` module containing stable `brotli` crate wrapper types.
- Add `BrotliEncoder::with_quality_and_params()` constructors.
- Add `Deflate64Decoder` behind new crate feature `deflate64`.

## 0.4.1 - 2023-07-10

- Add `Zstd{Encoder,Decoder}::with_dict()` constructors.
- Add `zstdmt` crate feature that enables `zstd-safe/zstdmt`, allowing multi-threaded functionality to work as expected.

## 0.4.0 - 2023-05-10

- `Level::Precise` variant now takes a `i32` instead of `u32`.
- Add top-level `zstd` module containing stable `zstd` crate wrapper types.
- Add `ZstdEncoder::with_quality_and_params()` constructors.
- Update `zstd` dependency to `0.12`.
- Remove deprecated `stream`, `futures-bufread` and `futures-write` crate features.
- Remove Tokio 0.2.x and 0.3.x support (`tokio-02` and `tokio-03` crate features).

## 0.3.15 - 2022-10-08

- `Level::Default::into_zstd()` now returns zstd's default value `3`.
- Fix endianness when reading the `extra` field of a gzip header.
