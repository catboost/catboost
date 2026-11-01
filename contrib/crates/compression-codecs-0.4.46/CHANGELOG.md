# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.46](https://github.com/Nullus157/async-compression/compare/compression-codecs-v0.4.45...compression-codecs-v0.4.46) - 2026-10-17

### Added

- *(zstd)* add `mzstd` v0.2.5 ([#505](https://github.com/Nullus157/async-compression/pull/505))

### Other

- add keywords to compression crates ([#502](https://github.com/Nullus157/async-compression/pull/502))

## [0.4.45](https://github.com/Nullus157/async-compression/compare/compression-codecs-v0.4.44...compression-codecs-v0.4.45) - 2026-09-30

### Other

- *(deps)* update brotli requirement from 8 to 9 ([#499](https://github.com/Nullus157/async-compression/pull/499))

## [0.4.44](https://github.com/Nullus157/async-compression/compare/compression-codecs-v0.4.43...compression-codecs-v0.4.44) - 2026-09-29

### Added

- *(brotli)* add optional mbrotli backend ([#492](https://github.com/Nullus157/async-compression/pull/492))

## [0.4.43](https://github.com/Nullus157/async-compression/compare/compression-codecs-v0.4.42...compression-codecs-v0.4.43) - 2026-09-16

### Fixed

- *(gzip)* reject invalid header before the full 10 bytes arrive ([#489](https://github.com/Nullus157/async-compression/pull/489))

## [0.4.42](https://github.com/Nullus157/async-compression/compare/compression-codecs-v0.4.41...compression-codecs-v0.4.42) - 2026-09-14

### Fixed

- *(zstd)* avoid no-progress errors while awaiting input ([#486](https://github.com/Nullus157/async-compression/pull/486))

## [0.4.41](https://github.com/Nullus157/async-compression/compare/compression-codecs-v0.4.40...compression-codecs-v0.4.41) - 2026-09-09

### Other

- Return UnexpectedEof when deflate64 does not progress ([#483](https://github.com/Nullus157/async-compression/pull/483))

## [0.4.40](https://github.com/Nullus157/async-compression/compare/compression-codecs-v0.4.39...compression-codecs-v0.4.40) - 2026-09-07

### Other

- *(deps)* update zstd requirement from 0.13 to 0.14 ([#481](https://github.com/Nullus157/async-compression/pull/481))

## [0.4.39](https://github.com/Nullus157/async-compression/compare/compression-codecs-v0.4.38...compression-codecs-v0.4.39) - 2026-09-02

### Other

- updated the following local packages: compression-core

## [0.4.38](https://github.com/Nullus157/async-compression/compare/compression-codecs-v0.4.37...compression-codecs-v0.4.38) - 2026-04-21

### Other

- Include license files in published crates ([#467](https://github.com/Nullus157/async-compression/pull/467))

## [0.4.37](https://github.com/Nullus157/async-compression/compare/compression-codecs-v0.4.36...compression-codecs-v0.4.37) - 2026-02-18

### Other

- Stop consuming input on errors in codecs. ([#451](https://github.com/Nullus157/async-compression/pull/451))

## [0.4.36](https://github.com/Nullus157/async-compression/compare/compression-codecs-v0.4.35...compression-codecs-v0.4.36) - 2026-01-10

### Other

- Set async-compression msrv ([#440](https://github.com/Nullus157/async-compression/pull/440))

## [0.4.35](https://github.com/Nullus157/async-compression/compare/compression-codecs-v0.4.34...compression-codecs-v0.4.35) - 2025-12-11

### Other

- Add Crc checksum validation for gzip::header::Parser ([#432](https://github.com/Nullus157/async-compression/pull/432))

## [0.4.34](https://github.com/Nullus157/async-compression/compare/compression-codecs-v0.4.33...compression-codecs-v0.4.34) - 2025-12-07

### Other

- Optimize GzipEncoder to not allocate for header and footer ([#431](https://github.com/Nullus157/async-compression/pull/431))
- Optimize GzipDecoder ([#430](https://github.com/Nullus157/async-compression/pull/430))
- Optimize gzip::header remove heap allocation ([#428](https://github.com/Nullus157/async-compression/pull/428))

## [0.4.33](https://github.com/Nullus157/async-compression/compare/compression-codecs-v0.4.32...compression-codecs-v0.4.33) - 2025-11-21

### Added

- Add uninitialized output buffer support ([#414](https://github.com/Nullus157/async-compression/pull/414))

### Other

- Set msrv for codecs to 1.83 ([#422](https://github.com/Nullus157/async-compression/pull/422))

## [0.4.32](https://github.com/Nullus157/async-compression/compare/compression-codecs-v0.4.31...compression-codecs-v0.4.32) - 2025-11-06

### Added

- allow reading uncompressed size ([#396](https://github.com/Nullus157/async-compression/pull/396))

### Fixed

- `UnexpectedEof` on truncated input ([#412](https://github.com/Nullus157/async-compression/pull/412))

### Other

- Simplify `WriteBuffer`: Rm `unwritten_initialized_mut` ([#413](https://github.com/Nullus157/async-compression/pull/413))
- Implement new traits `DecodeV2`/`EncodeV2` ([#398](https://github.com/Nullus157/async-compression/pull/398))
- Re-export core in codecs ([#395](https://github.com/Nullus157/async-compression/pull/395))
- Disable nightly feature `doc_auto_cfg` on docsrs ([#392](https://github.com/Nullus157/async-compression/pull/392))

## [0.4.31](https://github.com/Nullus157/async-compression/compare/compression-codecs-v0.4.30...compression-codecs-v0.4.31) - 2025-09-25

### Other

- Use io::ErrorKind::OutOfMemory for xz2/bzip2 codecs ([#387](https://github.com/Nullus157/async-compression/pull/387))

## [0.4.30](https://github.com/Nullus157/async-compression/compare/compression-codecs-v0.4.29...compression-codecs-v0.4.30) - 2025-08-31

### Other

- rm unused dep from async-compression and compression-codecs ([#381](https://github.com/Nullus157/async-compression/pull/381))

## [0.4.29](https://github.com/Nullus157/async-compression/compare/compression-codecs-v0.4.28...compression-codecs-v0.4.29) - 2025-08-28

### Other

- Update Deps.rs badge ([#380](https://github.com/Nullus157/async-compression/pull/380))
- move async-compression to crates/ ([#379](https://github.com/Nullus157/async-compression/pull/379))

## [0.4.28](https://github.com/Nullus157/async-compression/compare/compression-codecs-v0.4.27...compression-codecs-v0.4.28) - 2025-08-23

### Fixed

- fix wasi ci testing and update doc in README ([#367](https://github.com/Nullus157/async-compression/pull/367))

### Other

- Have separate package.version field for compression-* ([#369](https://github.com/Nullus157/async-compression/pull/369))
- Fix docs.rs build for compression-codecs ([#365](https://github.com/Nullus157/async-compression/pull/365))
