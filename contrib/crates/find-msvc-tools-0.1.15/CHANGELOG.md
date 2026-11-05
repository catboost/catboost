# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.15](https://github.com/rust-lang/cc-rs/compare/find-msvc-tools-v0.1.14...find-msvc-tools-v0.1.15) - 2026-10-19

### Other

- Regenerate windows sys bindings ([#2019](https://github.com/rust-lang/cc-rs/pull/2019))

## [0.1.14](https://github.com/rust-lang/cc-rs/compare/find-msvc-tools-v0.1.13...find-msvc-tools-v0.1.14) - 2026-09-25

### Fixed

- Make `windows_sys` more private and re-export types needed by cc-rs ([#1944](https://github.com/rust-lang/cc-rs/pull/1944))

## [0.1.13](https://github.com/rust-lang/cc-rs/compare/find-msvc-tools-v0.1.12...find-msvc-tools-v0.1.13) - 2026-09-18

### Other

- Regenerate windows sys bindings ([#1926](https://github.com/rust-lang/cc-rs/pull/1926) [#1919](https://github.com/rust-lang/cc-rs/pull/1919))

## [0.1.12](https://github.com/rust-lang/cc-rs/compare/find-msvc-tools-v0.1.11...find-msvc-tools-v0.1.12) - 2026-09-04

### Other

- simplify conditional logic using `then` method ([#1860](https://github.com/rust-lang/cc-rs/pull/1860))

## [0.1.11](https://github.com/rust-lang/cc-rs/compare/find-msvc-tools-v0.1.10...find-msvc-tools-v0.1.11) - 2026-08-14

### Other

- Update MSRV to 1.65 ([#1834](https://github.com/rust-lang/cc-rs/pull/1834))

## [0.1.10](https://github.com/rust-lang/cc-rs/compare/find-msvc-tools-v0.1.9...find-msvc-tools-v0.1.10) - 2026-08-07

### Other

- Update edition to 2021 ([#1811](https://github.com/rust-lang/cc-rs/pull/1811))
- Update MSRV to 1.64 ([#1808](https://github.com/rust-lang/cc-rs/pull/1808))

## [0.1.9](https://github.com/rust-lang/cc-rs/compare/find-msvc-tools-v0.1.8...find-msvc-tools-v0.1.9) - 2026-01-30

### Other

- Add `find_windows_sdk` API ([#1663](https://github.com/rust-lang/cc-rs/pull/1663))

## [0.1.8](https://github.com/rust-lang/cc-rs/compare/find-msvc-tools-v0.1.7...find-msvc-tools-v0.1.8) - 2026-01-16

### Other

- Regenerate windows sys bindings ([#1653](https://github.com/rust-lang/cc-rs/pull/1653))

## [0.1.7](https://github.com/rust-lang/cc-rs/compare/find-msvc-tools-v0.1.6...find-msvc-tools-v0.1.7) - 2026-01-09

### Other

- Fix tool existence check in find_tool method ([#1645](https://github.com/rust-lang/cc-rs/pull/1645))
- Fix SdkInfo::find_tool to check for executable extension ([#1644](https://github.com/rust-lang/cc-rs/pull/1644))

## [0.1.6](https://github.com/rust-lang/cc-rs/compare/find-msvc-tools-v0.1.5...find-msvc-tools-v0.1.6) - 2025-12-26

### Other

- Update Readmes ([#1641](https://github.com/rust-lang/cc-rs/pull/1641))

## [0.1.5](https://github.com/rust-lang/cc-rs/compare/find-msvc-tools-v0.1.4...find-msvc-tools-v0.1.5) - 2025-11-14

### Other

- Add Visual Studio 2026 support ([#1609](https://github.com/rust-lang/cc-rs/pull/1609))

## [0.1.4](https://github.com/rust-lang/cc-rs/compare/find-msvc-tools-v0.1.3...find-msvc-tools-v0.1.4) - 2025-10-10

### Other

- Allow using VCToolsVersion to request a specific msvc version ([#1589](https://github.com/rust-lang/cc-rs/pull/1589))
- Regenerate windows sys bindings ([#1591](https://github.com/rust-lang/cc-rs/pull/1591))

## [0.1.3](https://github.com/rust-lang/cc-rs/compare/find-msvc-tools-v0.1.2...find-msvc-tools-v0.1.3) - 2025-10-03

### Other

- Regenerate windows sys bindings ([#1572](https://github.com/rust-lang/cc-rs/pull/1572))

## [0.1.2](https://github.com/rust-lang/cc-rs/compare/find-msvc-tools-v0.1.1...find-msvc-tools-v0.1.2) - 2025-09-19

### Other

- [win] Search the Windows SDK for tools as well ([#1553](https://github.com/rust-lang/cc-rs/pull/1553))

## [0.1.1](https://github.com/rust-lang/cc-rs/compare/find-msvc-tools-v0.1.0...find-msvc-tools-v0.1.1) - 2025-09-05

### Other

- Regenerate windows sys bindings ([#1548](https://github.com/rust-lang/cc-rs/pull/1548))
- Add fn get_ucrt_dir for find-msvc-tools ([#1546](https://github.com/rust-lang/cc-rs/pull/1546))
