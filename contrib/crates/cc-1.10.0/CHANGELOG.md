# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.10.0](https://github.com/rust-lang/cc-rs/compare/cc-v1.9.0...cc-v1.10.0) - 2026-10-23

### Added

- Initial libtool support ([#1618](https://github.com/rust-lang/cc-rs/pull/1618))

### Fixed

- *(parallel)* print the child's exit status as debug output ([#2055](https://github.com/rust-lang/cc-rs/pull/2055))
- *(target-parsing)* wasm32-none-component ([#2054](https://github.com/rust-lang/cc-rs/pull/2054))

### Other

- Regenerate target info ([#2053](https://github.com/rust-lang/cc-rs/pull/2053))

## [1.9.0](https://github.com/rust-lang/cc-rs/compare/cc-v1.8.0...cc-v1.9.0) - 2026-10-19

### Added

- pass AddressSanitizer on to the C compiler ([#2045](https://github.com/rust-lang/cc-rs/pull/2045))
- add source root for stable external object names ([#2034](https://github.com/rust-lang/cc-rs/pull/2034))
- emit rerun-if-env-changed for the compiler's own env vars ([#2035](https://github.com/rust-lang/cc-rs/pull/2035))
- *(env)* read Cargo's variables from the env snapshot ([#2028](https://github.com/rust-lang/cc-rs/pull/2028))
- collect compile commands into a `compile_commands.json` ([#1990](https://github.com/rust-lang/cc-rs/pull/1990))

### Fixed

- don't prefer clang for -Clinker-plugin-lto=no ([#2046](https://github.com/rust-lang/cc-rs/pull/2046))
- keep the order of flag and flag_if_supported ([#2044](https://github.com/rust-lang/cc-rs/pull/2044))
- correct "arbitary" typo in target parser comment ([#2039](https://github.com/rust-lang/cc-rs/pull/2039))
- correct "stablised" typos in TODO comments ([#2040](https://github.com/rust-lang/cc-rs/pull/2040))
- *(msvc)* pass objects to the archiver relative to the output directory ([#2033](https://github.com/rust-lang/cc-rs/pull/2033))
- honor disabled linker-plugin LTO ([#2024](https://github.com/rust-lang/cc-rs/pull/2024))
- apply `RUSTC_WRAPPER` to the `cl.exe` cc finds itself ([#2022](https://github.com/rust-lang/cc-rs/pull/2022))

### Other

- *(parallel)* poll on jobserver helper thread ([#2021](https://github.com/rust-lang/cc-rs/pull/2021))
- *(parallel)* add jobserver polling for windows ([#2020](https://github.com/rust-lang/cc-rs/pull/2020))
- Regenerate windows sys bindings ([#2019](https://github.com/rust-lang/cc-rs/pull/2019))
- *(parallel)* add polling for process exit on windows ([#2016](https://github.com/rust-lang/cc-rs/pull/2016))
- *(parallel)* poll for process exit on macOS/bsd ([#2014](https://github.com/rust-lang/cc-rs/pull/2014))

## [1.8.0](https://github.com/rust-lang/cc-rs/compare/cc-v1.7.0...cc-v1.8.0) - 2026-10-11

### Added

- add `CC_PREFER_CLANG_CL_OVER_MSVC` to override `prefer_clang_cl_over_msvc` ([#2001](https://github.com/rust-lang/cc-rs/pull/2001))

### Fixed

- *(panic)* fix panic in code and replace with error when appropriate ([#2008](https://github.com/rust-lang/cc-rs/pull/2008))
- *(env)* do not panic if WASM_MUSL_SYSROOT not set ([#2007](https://github.com/rust-lang/cc-rs/pull/2007))
- *(jobserver)* help thread gets more token than necessary ([#2006](https://github.com/rust-lang/cc-rs/pull/2006))
- stop forwarding -Cprofile-generate/-Cprofile-use to the C compiler ([#2003](https://github.com/rust-lang/cc-rs/pull/2003))
- do not apply target crt-static when target is overridden ([#1999](https://github.com/rust-lang/cc-rs/pull/1999))

### Other

- *(parallel)* poll for process exit on linux ([#2009](https://github.com/rust-lang/cc-rs/pull/2009))
- document `NVCC` and other environment variables ([#2005](https://github.com/rust-lang/cc-rs/pull/2005))
- *(jobserver)* poll for new token ([#2004](https://github.com/rust-lang/cc-rs/pull/2004))
- *(parallel)* Use poll on unix ([#2000](https://github.com/rust-lang/cc-rs/pull/2000))

## [1.7.0](https://github.com/rust-lang/cc-rs/compare/cc-v1.6.0...cc-v1.7.0) - 2026-10-10

### Added

- add `Tool::get_envs`, deprecate Tool::env and store tool env as `Arc<OsStr>` pairs ([#1985](https://github.com/rust-lang/cc-rs/pull/1985))
- link `c++` when Clang uses libc++ ([#1978](https://github.com/rust-lang/cc-rs/pull/1978))
- add `CC_MASM_ASM` and fall back to `llvm-ml` for MASM files ([#1975](https://github.com/rust-lang/cc-rs/pull/1975))

### Fixed

- *(target parsing)* for powerpc64-ibm-aix abi ([#1989](https://github.com/rust-lang/cc-rs/pull/1989))
- ignore ASCII case in program name checks ([#1981](https://github.com/rust-lang/cc-rs/pull/1981))

### Other

- *(compile)* issue compilation cmds more frequently in parallel #1997 #1998
- *(cache)* Optimize tool detection caching ([#1995](https://github.com/rust-lang/cc-rs/pull/1995))
- *(cache)* optimize EnvSnapshot Hash, PartialEq, Ord impl ([#1994](https://github.com/rust-lang/cc-rs/pull/1994))
- Regenerate target info ([#1988](https://github.com/rust-lang/cc-rs/pull/1988))

## [1.6.0](https://github.com/rust-lang/cc-rs/compare/cc-v1.5.1...cc-v1.6.0) - 2026-10-03

### Added

- add `Build::create_archive` and `emit_link_directives` ([#1972](https://github.com/rust-lang/cc-rs/pull/1972))
- read the process environment once per `Build`, on first use ([#1969](https://github.com/rust-lang/cc-rs/pull/1969))
- add `Build::message_logger` for structured access to cc's messages ([#1967](https://github.com/rust-lang/cc-rs/pull/1967))
- add `CXXSTDLIB_STATIC` to link the C++ stdlib statically from outside ([#1957](https://github.com/rust-lang/cc-rs/pull/1957))

### Fixed

- key the flag support cache on target, host, language and environment ([#1965](https://github.com/rust-lang/cc-rs/pull/1965))
- forward each line of compiler stderr once and whole with `parallel` ([#1959](https://github.com/rust-lang/cc-rs/pull/1959))

### Other

- find compile commands by source file so tests pass with `parallel` ([#1973](https://github.com/rust-lang/cc-rs/pull/1973))
- add support for `armeb-unknown-linux-gnueabi` ([#1958](https://github.com/rust-lang/cc-rs/pull/1958))

## [1.5.1](https://github.com/rust-lang/cc-rs/compare/cc-v1.5.0...cc-v1.5.1) - 2026-09-25

### Fixed

- link the static C++ stdlib with `-bundle`, once per `Build` ([#1955](https://github.com/rust-lang/cc-rs/pull/1955))

### Other

- Use lp64d ABI for Redox riscv64 ([#1953](https://github.com/rust-lang/cc-rs/pull/1953))

## [1.5.0](https://github.com/rust-lang/cc-rs/compare/cc-v1.4.7...cc-v1.5.0) - 2026-09-25

### Added

- inherit x86 target features from RUSTFLAGS ([#1948](https://github.com/rust-lang/cc-rs/pull/1948))

### Fixed

- don't report the file name cl.exe echoes as a warning in expand() ([#1950](https://github.com/rust-lang/cc-rs/pull/1950))
- ignore MSVC `/link` flags with a warning ([#1949](https://github.com/rust-lang/cc-rs/pull/1949))
- Make `windows_sys` more private and re-export types needed by cc-rs ([#1944](https://github.com/rust-lang/cc-rs/pull/1944))
- pass -Tp on MSVC for .cc so they are not treated as objects ([#1930](https://github.com/rust-lang/cc-rs/pull/1930))

### Other

- document macOS SDKROOT and Xcode CLT for testing ([#1943](https://github.com/rust-lang/cc-rs/pull/1943))
- remove License section from CONTRIBUTING.md ([#1942](https://github.com/rust-lang/cc-rs/pull/1942))
- add CONTRIBUTING.md and Conventional Commit PR title check ([#1938](https://github.com/rust-lang/cc-rs/pull/1938))

### Fixed

- Pass `-Tp` immediately before `.cc` sources on MSVC (not clang-cl) when compiling C++ so they are not treated as object files ([#1877](https://github.com/rust-lang/cc-rs/issues/1877))

## [1.4.7](https://github.com/rust-lang/cc-rs/compare/cc-v1.4.6...cc-v1.4.7) - 2026-09-18

### Fixed

- strip OUT_DIR from the object file name hash too ([#1902](https://github.com/rust-lang/cc-rs/pull/1902))
- search_is_some clippy lint ([#1903](https://github.com/rust-lang/cc-rs/pull/1903))

### Other

- Regenerate target info ([#1924](https://github.com/rust-lang/cc-rs/pull/1924))
- Regenerate windows sys bindings ([#1919](https://github.com/rust-lang/cc-rs/pull/1919))
- Fix target info parsing ([#1911](https://github.com/rust-lang/cc-rs/pull/1911))
- Add Z80 and SM83 target support ([#1900](https://github.com/rust-lang/cc-rs/pull/1900))

## [1.4.6](https://github.com/rust-lang/cc-rs/compare/cc-v1.4.5...cc-v1.4.6) - 2026-09-11

### Fixed

- Prevent check_exe from dropping flags on the floor. ([#1884](https://github.com/rust-lang/cc-rs/pull/1884))
- update fd handling to use BorrowedFd ([#1882](https://github.com/rust-lang/cc-rs/pull/1882))

### Other

- *(deps)* bump taiki-e/install-action from 2.87.2 to 2.87.4 ([#1888](https://github.com/rust-lang/cc-rs/pull/1888))
- Regenerate target info ([#1891](https://github.com/rust-lang/cc-rs/pull/1891))
- *(deps)* bump tombi-toml/setup-tombi from 1.5.0 to 1.5.1 ([#1890](https://github.com/rust-lang/cc-rs/pull/1890))
- *(deps)* bump release-plz/action from 0.5.131 to 0.5.132 ([#1889](https://github.com/rust-lang/cc-rs/pull/1889))
- Fix ci(test): make sure right toolchain is passed ([#1892](https://github.com/rust-lang/cc-rs/pull/1892))
- *(deps)* bump tombi-toml/setup-tombi from 1.2.5 to 1.5.0 ([#1885](https://github.com/rust-lang/cc-rs/pull/1885))
- *(deps)* bump taiki-e/install-action from 2.85.5 to 2.87.2 ([#1886](https://github.com/rust-lang/cc-rs/pull/1886))
- add toml formatting check ([#1879](https://github.com/rust-lang/cc-rs/pull/1879))

## [1.4.5](https://github.com/rust-lang/cc-rs/compare/cc-v1.4.4...cc-v1.4.5) - 2026-09-04

### Fixed

- probe flag support without OUT_DIR via tempfile ([#1875](https://github.com/rust-lang/cc-rs/pull/1875))

### Other

- simplify conditional logic using `then` method ([#1860](https://github.com/rust-lang/cc-rs/pull/1860))

### Fixed

- Probe flag support without `OUT_DIR` via tempfile, so `flag_if_supported` no longer silently drops flags outside Cargo build scripts ([#1875](https://github.com/rust-lang/cc-rs/pull/1875))

## [1.4.4](https://github.com/rust-lang/cc-rs/compare/cc-v1.4.3...cc-v1.4.4) - 2026-08-21

### Fixed

- honour `Build::env` overrides in Android llvm-ar probe ([#1868](https://github.com/rust-lang/cc-rs/pull/1868))
- honour `Build::env` overrides in flag support and compiler family detection probes ([#1866](https://github.com/rust-lang/cc-rs/pull/1866))

## [1.4.3](https://github.com/rust-lang/cc-rs/compare/cc-v1.4.2...cc-v1.4.3) - 2026-08-14

### Other

- Update MSRV to 1.65 ([#1834](https://github.com/rust-lang/cc-rs/pull/1834))
- Regenerate target info ([#1848](https://github.com/rust-lang/cc-rs/pull/1848))

## [1.4.2](https://github.com/rust-lang/cc-rs/compare/cc-v1.4.1...cc-v1.4.2) - 2026-08-08

### Fixed

- Infer NEON, not VFPv4, from `neon` in the target name ([#1843](https://github.com/rust-lang/cc-rs/pull/1843))
- do not emit `-mno-omit-leaf-frame-pointer` if unsupported ([#1845](https://github.com/rust-lang/cc-rs/pull/1845))

## [1.4.1](https://github.com/rust-lang/cc-rs/compare/cc-v1.4.0...cc-v1.4.1) - 2026-08-07

### Fixed

- Fix `-Cforce-frame-pointers`'s corresponding `CFLAGS` ([#1814](https://github.com/rust-lang/cc-rs/pull/1814))
- Fix parsing of thumbv7a-vex-v5 ([#1840](https://github.com/rust-lang/cc-rs/pull/1840))

### Other

- Regenerate target info ([#1839](https://github.com/rust-lang/cc-rs/pull/1839))
- Add `aarch64_be` to `prefix_for_target` ([#1835](https://github.com/rust-lang/cc-rs/pull/1835))
- Use lp64d ABI for Managarm riscv64 ([#1829](https://github.com/rust-lang/cc-rs/pull/1829))
- Make cc work on Motor OS ([#1822](https://github.com/rust-lang/cc-rs/pull/1822))
- Update edition to 2021 ([#1811](https://github.com/rust-lang/cc-rs/pull/1811))
- Update MSRV to 1.64 ([#1808](https://github.com/rust-lang/cc-rs/pull/1808))

## [1.4.0](https://github.com/rust-lang/cc-rs/compare/cc-v1.3.0...cc-v1.4.0) - 2026-07-24

### Added

- support trim paths with clang-cl.exe ([#1799](https://github.com/rust-lang/cc-rs/pull/1799))

## [1.3.0](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.67...cc-v1.3.0) - 2026-07-18

### Added

- inherit path remap rules from cargo trim-paths ([#1794](https://github.com/rust-lang/cc-rs/pull/1794))

## [1.2.67](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.66...cc-v1.2.67) - 2026-07-11

### Other

- Fix clippy warning ([#1788](https://github.com/rust-lang/cc-rs/pull/1788))
- Regenerate target info ([#1785](https://github.com/rust-lang/cc-rs/pull/1785))
- Add support for `aarch64-unknown-linux-pauthtest` target ([#1713](https://github.com/rust-lang/cc-rs/pull/1713))
- Fix nightly compilation error ([#1783](https://github.com/rust-lang/cc-rs/pull/1783))

## [1.2.66](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.65...cc-v1.2.66) - 2026-07-05

### Other

- Fix target parsing for aarch64-unknown-linux-pauthtest ([#1779](https://github.com/rust-lang/cc-rs/pull/1779))
- Support new QNX targets ([#1775](https://github.com/rust-lang/cc-rs/pull/1775))
- Add kache to the supported compiler wrappers ([#1770](https://github.com/rust-lang/cc-rs/pull/1770))

## [1.2.65](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.64...cc-v1.2.65) - 2026-06-19

### Other

- Regenerate target info ([#1763](https://github.com/rust-lang/cc-rs/pull/1763))

## [1.2.64](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.63...cc-v1.2.64) - 2026-06-12

### Other

- Add more bare-metal 32-bit Arm support ([#1753](https://github.com/rust-lang/cc-rs/pull/1753))
- Remove unnecessary clone ([#1748](https://github.com/rust-lang/cc-rs/pull/1748))
- Add default compiler prefixes for thumbv7a/thumbv7r/thumbv8r bare-metal targets ([#1742](https://github.com/rust-lang/cc-rs/pull/1742))

## [1.2.63](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.62...cc-v1.2.63) - 2026-05-29

### Other

- Update shlex requirement from 1.3.0 to 2.0.1 ([#1736](https://github.com/rust-lang/cc-rs/pull/1736))

## [1.2.62](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.61...cc-v1.2.62) - 2026-05-08

### Other

- Regenerate target info ([#1721](https://github.com/rust-lang/cc-rs/pull/1721))
- Allow exceptions on wasm platforms ([#1714](https://github.com/rust-lang/cc-rs/pull/1714))
- Add relibc env ([#1710](https://github.com/rust-lang/cc-rs/pull/1710))
- recognize sh4 architecture in parse_arch() ([#1712](https://github.com/rust-lang/cc-rs/pull/1712))

## [1.2.61](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.60...cc-v1.2.61) - 2026-04-24

### Other

- fix `OutputKind::Capture` documentation ([#1705](https://github.com/rust-lang/cc-rs/pull/1705))

## [1.2.60](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.59...cc-v1.2.60) - 2026-04-10

### Fixed

- *(ar)* suppress warnings from `D` modifier probe ([#1700](https://github.com/rust-lang/cc-rs/pull/1700))

## [1.2.59](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.58...cc-v1.2.59) - 2026-04-03

### Fixed

- *(ar)* deterministic archives with `D` modifier ([#1697](https://github.com/rust-lang/cc-rs/pull/1697))

### Other

- Regenerate target info ([#1698](https://github.com/rust-lang/cc-rs/pull/1698))
- Fix target abi parsing for sanitiser targets ([#1695](https://github.com/rust-lang/cc-rs/pull/1695))

## [1.2.58](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.57...cc-v1.2.58) - 2026-03-27

### Other

- Update Compile-time Requirements to add info about clang-cl.exe ([#1693](https://github.com/rust-lang/cc-rs/pull/1693))

## [1.2.57](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.56...cc-v1.2.57) - 2026-03-13

### Other

- Size archiver batches according to argument length not argument count ([#1689](https://github.com/rust-lang/cc-rs/pull/1689))
- Added `Build::env` for setting environment variables of compiler invocations and other child processes ([#1656](https://github.com/rust-lang/cc-rs/pull/1656) [#1682](https://github.com/rust-lang/cc-rs/pull/1682))

## [1.2.56](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.55...cc-v1.2.56) - 2026-02-13

### Other

- Regenerate target info ([#1676](https://github.com/rust-lang/cc-rs/pull/1676))
- Fix `clang-cl` target when cross-compiling ([#1670](https://github.com/rust-lang/cc-rs/pull/1670))

## [1.2.55](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.54...cc-v1.2.55) - 2026-01-30

### Other

- Regenerate target info ([#1667](https://github.com/rust-lang/cc-rs/pull/1667))
- Fix RUSTFLAGS typo in test-linker-plugin-lto ([#1665](https://github.com/rust-lang/cc-rs/pull/1665))
- Disable PIC for armv7-sony-vita-newlibeabihf ([#1664](https://github.com/rust-lang/cc-rs/pull/1664))

## [1.2.54](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.53...cc-v1.2.54) - 2026-01-23

### Other

- Fix x86_64-unknown-linux-gnuasan parsing ([#1661](https://github.com/rust-lang/cc-rs/pull/1661))
- Regenerate target info ([#1660](https://github.com/rust-lang/cc-rs/pull/1660))

## [1.2.53](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.52...cc-v1.2.53) - 2026-01-16

### Other

- Add missing RISC-V targets ([#1657](https://github.com/rust-lang/cc-rs/pull/1657))

## [1.2.52](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.51...cc-v1.2.52) - 2026-01-09

### Other

- Fix contradictory doc for CC compiler in crate doc ([#1650](https://github.com/rust-lang/cc-rs/pull/1650))
- Have CUDA compilaion check for sbsa-linux when targeting aarch64. ([#1647](https://github.com/rust-lang/cc-rs/pull/1647))
- Update link for -Cdwarf-version; Remove -Z (stabilized in 1.88) ([#1648](https://github.com/rust-lang/cc-rs/pull/1648))
- Fix Build::env_tool to check for .exe on windows ([#1646](https://github.com/rust-lang/cc-rs/pull/1646))

## [1.2.51](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.50...cc-v1.2.51) - 2025-12-26

### Other

- Regenerate target info ([#1642](https://github.com/rust-lang/cc-rs/pull/1642))
- Update Readmes ([#1641](https://github.com/rust-lang/cc-rs/pull/1641))

## [1.2.50](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.49...cc-v1.2.50) - 2025-12-19

### Other

- Add tests for `OUT_DIR` escape for '..' file paths (#1631)
- Fix #283: Make warnings(false) actually suppress compiler warnings ([#1633](https://github.com/rust-lang/cc-rs/pull/1633))

## [1.2.49](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.48...cc-v1.2.49) - 2025-12-06

### Other

- Fix run_output to prevent infinite blocking ([#1627](https://github.com/rust-lang/cc-rs/pull/1627))
- Fix detect_family deadlock ([#1626](https://github.com/rust-lang/cc-rs/pull/1626))
- Fix link in new debug_str doc comment ([#1625](https://github.com/rust-lang/cc-rs/pull/1625))
- Support more of Cargo's debug levels with Build::debug_str ([#1624](https://github.com/rust-lang/cc-rs/pull/1624))

## [1.2.48](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.47...cc-v1.2.48) - 2025-11-28

### Other

- Regenerate target info ([#1620](https://github.com/rust-lang/cc-rs/pull/1620))

## [1.2.47](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.46...cc-v1.2.47) - 2025-11-21

### Other

- add helenos linker identifications ([#1615](https://github.com/rust-lang/cc-rs/pull/1615))

## [1.2.46](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.45...cc-v1.2.46) - 2025-11-14

### Other

- Add Visual Studio 2026 support ([#1609](https://github.com/rust-lang/cc-rs/pull/1609))

## [1.2.45](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.44...cc-v1.2.45) - 2025-11-07

### Other

- Regenerate target info ([#1606](https://github.com/rust-lang/cc-rs/pull/1606))
- Use a default check for the "env" variable in apple_sdk_name ([#1605](https://github.com/rust-lang/cc-rs/pull/1605))

## [1.2.44](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.43...cc-v1.2.44) - 2025-10-31

### Other

- Fix debug assertion for env/abi mismatch ([#1604](https://github.com/rust-lang/cc-rs/pull/1604))
- Update CHANGELOG for version 1.2.43 ([#1602](https://github.com/rust-lang/cc-rs/pull/1602))
- Stop passing an invalid target to `llvm-mingw`'s cross-compilation wrappers ([#1495](https://github.com/rust-lang/cc-rs/pull/1495))

## [1.2.43](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.42...cc-v1.2.43) - 2025-10-24

### Other

- Mark `static_flag` and `shared_flag` as deprecated ([#1582](https://github.com/rust-lang/cc-rs/pull/1582))

## [1.2.42](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.41...cc-v1.2.42) - 2025-10-24

### Other

- Fix check-semver-checks ([#1600](https://github.com/rust-lang/cc-rs/pull/1600))
- minor improvement for docs ([#1598](https://github.com/rust-lang/cc-rs/pull/1598))
- Fix linker-plugin-lto: use `-flto=thin` ([#1594](https://github.com/rust-lang/cc-rs/pull/1594))
- Disable check-buildstd for armv7k-apple-watchos ([#1599](https://github.com/rust-lang/cc-rs/pull/1599))
- Add elf abi to ppc64 targets ([#1596](https://github.com/rust-lang/cc-rs/pull/1596))

## [1.2.41](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.40...cc-v1.2.41) - 2025-10-10

### Other

- Allow using VCToolsVersion to request a specific msvc version ([#1589](https://github.com/rust-lang/cc-rs/pull/1589))
- Regenerate target info ([#1592](https://github.com/rust-lang/cc-rs/pull/1592))
- Regenerate windows sys bindings ([#1591](https://github.com/rust-lang/cc-rs/pull/1591))
- Update windows-bindgen requirement from 0.64 to 0.65 ([#1590](https://github.com/rust-lang/cc-rs/pull/1590))
- Fix `get_base_archiver_variant` for clang-cl: use `--print-search-dirs` ([#1587](https://github.com/rust-lang/cc-rs/pull/1587))

## [1.2.40](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.39...cc-v1.2.40) - 2025-10-03

### Other

- Reorder changelog and remove duplicate Unreleased section ([#1579](https://github.com/rust-lang/cc-rs/pull/1579))
- Prefer clang if linker-plugin-lto specified ([#1573](https://github.com/rust-lang/cc-rs/pull/1573))
- Fix building for Mac Catalyst ([#1577](https://github.com/rust-lang/cc-rs/pull/1577))
- Improve ESP microcontroller targets ([#1574](https://github.com/rust-lang/cc-rs/pull/1574))

## [1.2.39](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.38...cc-v1.2.39) - 2025-09-26

### Other

- Fix cross compilation to xtensa-esp32s3-espidf ([#1569](https://github.com/rust-lang/cc-rs/pull/1569))
- Fix autodetect_wasi_compiler: support non utf-8 path ([#1568](https://github.com/rust-lang/cc-rs/pull/1568))
- Regenerate target info ([#1567](https://github.com/rust-lang/cc-rs/pull/1567))
- Fix rustcflags mapping: require -Clinker-plugin-lto for -flto ([#1564](https://github.com/rust-lang/cc-rs/pull/1564))
- Use `$WASI_SDK_PATH` on WASI targets by default ([#1562](https://github.com/rust-lang/cc-rs/pull/1562))
- Fix atomicity violations in concurrent cache operations ([#1559](https://github.com/rust-lang/cc-rs/pull/1559))

## [1.2.38](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.37...cc-v1.2.38) - 2025-09-19

### Other

- updated the following local packages: find-msvc-tools

## [1.2.37](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.36...cc-v1.2.37) - 2025-09-12

### Other

- Fix errmsg in RustcCodegenFlags::set_rustc_flag ([#1551](https://github.com/rust-lang/cc-rs/pull/1551))
- propagate stack protector to Linux C compilers ([#1550](https://github.com/rust-lang/cc-rs/pull/1550))
- Extract new fn `run_commands_in_parallel` ([#1549](https://github.com/rust-lang/cc-rs/pull/1549))

## [1.2.36](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.35...cc-v1.2.36) - 2025-09-05

### Other

- Regenerate windows sys bindings ([#1548](https://github.com/rust-lang/cc-rs/pull/1548))
- Update windows-bindgen requirement from 0.62 to 0.63 ([#1547](https://github.com/rust-lang/cc-rs/pull/1547))
- Add fn get_ucrt_dir for find-msvc-tools ([#1546](https://github.com/rust-lang/cc-rs/pull/1546))
- Regenerate target info ([#1544](https://github.com/rust-lang/cc-rs/pull/1544))
- fix publish.yml ([#1543](https://github.com/rust-lang/cc-rs/pull/1543))
- Replace periods with underscores as well when parsing env variables ([#1541](https://github.com/rust-lang/cc-rs/pull/1541))

## [1.2.35](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.34...cc-v1.2.35) - 2025-09-01

### Fixed

- fix building for aarch64-apple-visionos-sim on nightly ([#1534](https://github.com/rust-lang/cc-rs/pull/1534))
- fix tests apple_sdkroot_wrong ([#1530](https://github.com/rust-lang/cc-rs/pull/1530))

### Other

- Regenerate target info ([#1536](https://github.com/rust-lang/cc-rs/pull/1536))
- Optimize Tool::to_command ([#1535](https://github.com/rust-lang/cc-rs/pull/1535))
- Extract find-msvc-tools ([#1531](https://github.com/rust-lang/cc-rs/pull/1531))
- Add prefer_clang_cl_over_msvc ([#1516](https://github.com/rust-lang/cc-rs/pull/1516))

## [1.2.34](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.33...cc-v1.2.34) - 2025-08-22

### Fixed

- add `-mcpu=mvp` and `-mmutable-globals` for `wasm32v1-none` ([#1524](https://github.com/rust-lang/cc-rs/pull/1524))

### Other

- Optimize parse_version in find_tools.rs ([#1527](https://github.com/rust-lang/cc-rs/pull/1527))
- Fallback to manually searching for tool dir ([#1526](https://github.com/rust-lang/cc-rs/pull/1526))

## [1.2.33](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.32...cc-v1.2.33) - 2025-08-15

### Other

- Regenerate target info ([#1521](https://github.com/rust-lang/cc-rs/pull/1521))
- [win][arm64ec] Add testing for Arm64EC Windows ([#1512](https://github.com/rust-lang/cc-rs/pull/1512))
- Fix parsing of nigthly targets ([#1517](https://github.com/rust-lang/cc-rs/pull/1517))
- [win][arm64ec] Fix finding assembler and setting is_arm for Arm64EC ([#1511](https://github.com/rust-lang/cc-rs/pull/1511))

## [1.2.32](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.31...cc-v1.2.32) - 2025-08-08

### Fixed

- fix new clippy lint introduced in rust 1.89.0 ([#1509](https://github.com/rust-lang/cc-rs/pull/1509))

### Other

- clarify cargo default if no rerun emitted ([#1508](https://github.com/rust-lang/cc-rs/pull/1508))
- extract compile_objects_sequential ([#1507](https://github.com/rust-lang/cc-rs/pull/1507))
- Windows `find_tools`: add support for finding Clang ([#1506](https://github.com/rust-lang/cc-rs/pull/1506))
- Add m68k-unknown-linux-gnu cross-compile target ([#1505](https://github.com/rust-lang/cc-rs/pull/1505))

## [1.2.31](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.30...cc-v1.2.31) - 2025-08-01

### Other

- Add doc for using sccache/ccache etc ([#1502](https://github.com/rust-lang/cc-rs/pull/1502))
- ability to statically link against C++ stdlib ([#1497](https://github.com/rust-lang/cc-rs/pull/1497))
- Add instructions on using sccache ([#1503](https://github.com/rust-lang/cc-rs/pull/1503))
- Add support for recognizing some architectures supported by GCC, but not LLVM. ([#1500](https://github.com/rust-lang/cc-rs/pull/1500))

## [1.2.30](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.29...cc-v1.2.30) - 2025-07-18

### Other

- define _REENTRANT by default ([#1496](https://github.com/rust-lang/cc-rs/pull/1496))

## [1.2.29](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.28...cc-v1.2.29) - 2025-07-05

### Other

- Fix target parsing for powerpc ([#1490](https://github.com/rust-lang/cc-rs/pull/1490))

## [1.2.28](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.27...cc-v1.2.28) - 2025-07-04

### Other

- Recognize `mlibc` environment ([#1488](https://github.com/rust-lang/cc-rs/pull/1488))
- Fix clippy warnings about not using variables in `format!` strings ([#1489](https://github.com/rust-lang/cc-rs/pull/1489))

## [1.2.27](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.26...cc-v1.2.27) - 2025-06-13

### Other

- Regenerate windows sys bindings ([#1485](https://github.com/rust-lang/cc-rs/pull/1485))
- Update windows-bindgen requirement from 0.61 to 0.62 ([#1484](https://github.com/rust-lang/cc-rs/pull/1484))
- Regenerate target info ([#1483](https://github.com/rust-lang/cc-rs/pull/1483))

## [1.2.26](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.25...cc-v1.2.26) - 2025-06-06

### Other

- Also set `SDKROOT` when building apple platforms ([#1475](https://github.com/rust-lang/cc-rs/pull/1475))
- use windows 2022 in CI ([#1479](https://github.com/rust-lang/cc-rs/pull/1479))
- Detect -Wslash-u-filename warning on clang-cl ([#1477](https://github.com/rust-lang/cc-rs/pull/1477))

## [1.2.25](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.24...cc-v1.2.25) - 2025-05-30

### Other

- make `powerp64` use `powerpc64-linux-gnu` prefix ([#1474](https://github.com/rust-lang/cc-rs/pull/1474))

## [1.2.24](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.23...cc-v1.2.24) - 2025-05-23

### Other

- Regenerate windows sys bindings ([#1471](https://github.com/rust-lang/cc-rs/pull/1471))

## [1.2.23](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.22...cc-v1.2.23) - 2025-05-16

### Other

- support "vxworks" and "nto" OSes on `get_base_archiver_variant` ([#1456](https://github.com/rust-lang/cc-rs/pull/1456))

## [1.2.22](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.21...cc-v1.2.22) - 2025-05-09

### Other

- Add `flags` method to `cc::Build` for adding multiple flags ([#1466](https://github.com/rust-lang/cc-rs/pull/1466))

## [1.2.21](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.20...cc-v1.2.21) - 2025-05-02

### Other

- Fix wasm32-unknown-unknown by passing -c ([#1424](https://github.com/rust-lang/cc-rs/pull/1424))

## [1.2.20](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.19...cc-v1.2.20) - 2025-04-25

### Other

- Regenerate target info ([#1461](https://github.com/rust-lang/cc-rs/pull/1461))
- Fix parser.rs on latest rustc nightly ([#1459](https://github.com/rust-lang/cc-rs/pull/1459))

## [1.2.19](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.18...cc-v1.2.19) - 2025-04-11

### Other

- Fix musl compilation: Add musl as a prefix fallback ([#1455](https://github.com/rust-lang/cc-rs/pull/1455))

## [1.2.18](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.17...cc-v1.2.18) - 2025-04-04

### Other

- Regenerate target info ([#1450](https://github.com/rust-lang/cc-rs/pull/1450))
- Use `std::thread::available_parallelism` for determining the default number of jobs ([#1447](https://github.com/rust-lang/cc-rs/pull/1447))
- Fix mips64-openwrt-linux-musl parsing ([#1449](https://github.com/rust-lang/cc-rs/pull/1449))
- Use compiler prefix `x86_64-linux-musl` ([#1443](https://github.com/rust-lang/cc-rs/pull/1443))

## [1.2.17](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.16...cc-v1.2.17) - 2025-03-21

### Other

- Regenerate target info ([#1439](https://github.com/rust-lang/cc-rs/pull/1439))
- Regenerate windows sys bindings ([#1437](https://github.com/rust-lang/cc-rs/pull/1437))
- Fix wasm32-wali-linux-musl target parsing ([#1434](https://github.com/rust-lang/cc-rs/pull/1434))
- Parse `rustc` target names ([#1413](https://github.com/rust-lang/cc-rs/pull/1413))
- Regenerate target info ([#1429](https://github.com/rust-lang/cc-rs/pull/1429))
- Added base support for `wasm32-wali-linux-musl` target ([#1373](https://github.com/rust-lang/cc-rs/pull/1373))

## [1.2.16](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.15...cc-v1.2.16) - 2025-02-28

### Fixed

- force windows compiler to run in `out_dir` to prevent artifacts in cwd (#1415)

### Other

- use `/arch:SSE2` for `x86` target arch (#1425)
- Regenerate windows-sys binding ([#1422](https://github.com/rust-lang/cc-rs/pull/1422))
- Regenerate target info ([#1418](https://github.com/rust-lang/cc-rs/pull/1418))
- Add LIB var when compiling flag_check (#1417)
- Change flag ordering ([#1403](https://github.com/rust-lang/cc-rs/pull/1403))
- Fix archiver detection for musl cross compilation ([#1404](https://github.com/rust-lang/cc-rs/pull/1404))

## [1.2.15](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.14...cc-v1.2.15) - 2025-02-21

### Other

- Regenerate target info ([#1406](https://github.com/rust-lang/cc-rs/pull/1406))
- Always read from all `CFLAGS`-style flags ([#1401](https://github.com/rust-lang/cc-rs/pull/1401))
- Simplify the error output on failed `Command` invocation ([#1397](https://github.com/rust-lang/cc-rs/pull/1397))

## [1.2.14](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.13...cc-v1.2.14) - 2025-02-14

### Other

- Regenerate target info ([#1398](https://github.com/rust-lang/cc-rs/pull/1398))
- Add support for setting `-gdwarf-{version}` based on RUSTFLAGS ([#1395](https://github.com/rust-lang/cc-rs/pull/1395))
- Add support for alternative network stack io-sock on QNX 7.1 aarch64 and x86_64 ([#1312](https://github.com/rust-lang/cc-rs/pull/1312))

## [1.2.13](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.12...cc-v1.2.13) - 2025-02-08

### Other

- Fix cross-compiling for Apple platforms ([#1389](https://github.com/rust-lang/cc-rs/pull/1389))

## [1.2.12](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.11...cc-v1.2.12) - 2025-02-04

### Other

- Split impl Build ([#1382](https://github.com/rust-lang/cc-rs/pull/1382))
- Don't specify both `-target` and `-mtargetos=` on Apple targets ([#1384](https://github.com/rust-lang/cc-rs/pull/1384))

## [1.2.11](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.10...cc-v1.2.11) - 2025-01-31

### Other

- Fix more flag inheritance ([#1380](https://github.com/rust-lang/cc-rs/pull/1380))
- Include wrapper args. in `stdout` family heuristics to restore classifying `clang --driver-mode=cl` as `Msvc { clang_cl: true }` ([#1378](https://github.com/rust-lang/cc-rs/pull/1378))
- Constrain `-Clto` and `-Cembed-bitcode` flag inheritance to be `clang`-only ([#1379](https://github.com/rust-lang/cc-rs/pull/1379))
- Pass deployment target with `-m*-version-min=` ([#1339](https://github.com/rust-lang/cc-rs/pull/1339))
- Regenerate target info ([#1376](https://github.com/rust-lang/cc-rs/pull/1376))

## [1.2.10](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.9...cc-v1.2.10) - 2025-01-17

### Other

- Fix CC_FORCE_DISABLE=0 evaluating to true ([#1371](https://github.com/rust-lang/cc-rs/pull/1371))
- Regenerate target info ([#1369](https://github.com/rust-lang/cc-rs/pull/1369))
- Make hidden lifetimes explicit. ([#1366](https://github.com/rust-lang/cc-rs/pull/1366))

## [1.2.9](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.8...cc-v1.2.9) - 2025-01-12

### Other

- Don't pass inherited PGO flags to GNU compilers (#1363)
- Adjusted zig cc judgment and avoided zigbuild errors([#1360](https://github.com/rust-lang/cc-rs/pull/1360)) ([#1361](https://github.com/rust-lang/cc-rs/pull/1361))
- Fix compilation on macOS using clang and fix compilation using zig-cc ([#1364](https://github.com/rust-lang/cc-rs/pull/1364))

## [1.2.8](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.7...cc-v1.2.8) - 2025-01-11

### Other

- Add `is_like_clang_cl()` getter (#1357)
- Fix clippy error in lib.rs ([#1356](https://github.com/rust-lang/cc-rs/pull/1356))
- Regenerate target info ([#1352](https://github.com/rust-lang/cc-rs/pull/1352))
- Fix compiler family detection issue with clang-cl on macOS ([#1328](https://github.com/rust-lang/cc-rs/pull/1328))
- Update `windows-bindgen` dependency ([#1347](https://github.com/rust-lang/cc-rs/pull/1347))
- Fix clippy warnings ([#1346](https://github.com/rust-lang/cc-rs/pull/1346))

## [1.2.7](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.6...cc-v1.2.7) - 2025-01-03

### Other

- Regenerate target info ([#1342](https://github.com/rust-lang/cc-rs/pull/1342))
- Document new supported architecture names in windows::find
- Make is_flag_supported_inner take an &Tool ([#1337](https://github.com/rust-lang/cc-rs/pull/1337))
- Fix is_flag_supported on msvc ([#1336](https://github.com/rust-lang/cc-rs/pull/1336))
- Allow using Visual Studio target names in `find_tool` ([#1335](https://github.com/rust-lang/cc-rs/pull/1335))

## [1.2.6](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.5...cc-v1.2.6) - 2024-12-27

### Other

- Don't inherit the `/Oy` flag for 64-bit targets ([#1330](https://github.com/rust-lang/cc-rs/pull/1330))

## [1.2.5](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.4...cc-v1.2.5) - 2024-12-19

### Other

- Check linking when testing if compiler flags are supported ([#1322](https://github.com/rust-lang/cc-rs/pull/1322))

## [1.2.4](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.3...cc-v1.2.4) - 2024-12-13

### Other

- Add support for C/C++ compiler for Neutrino QNX: `qcc` ([#1319](https://github.com/rust-lang/cc-rs/pull/1319))
- use -maix64 instead of -m64 ([#1307](https://github.com/rust-lang/cc-rs/pull/1307))

## [1.2.3](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.2...cc-v1.2.3) - 2024-12-06

### Other

- Improve detection of environment when compiling from msbuild or msvc ([#1310](https://github.com/rust-lang/cc-rs/pull/1310))
- Better error message when failing on unknown targets ([#1313](https://github.com/rust-lang/cc-rs/pull/1313))
- Optimize RustcCodegenFlags ([#1305](https://github.com/rust-lang/cc-rs/pull/1305))

## [1.2.2](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.1...cc-v1.2.2) - 2024-11-29

### Other

- Inherit flags from rustc ([#1279](https://github.com/rust-lang/cc-rs/pull/1279))
- Add support for using sccache wrapper with cuda/nvcc ([#1304](https://github.com/rust-lang/cc-rs/pull/1304))
- Fix msvc stdout not shown on error ([#1303](https://github.com/rust-lang/cc-rs/pull/1303))
- Regenerate target info ([#1301](https://github.com/rust-lang/cc-rs/pull/1301))
- Fix compilation of C++ code for armv7-unknown-linux-gnueabihf ([#1298](https://github.com/rust-lang/cc-rs/pull/1298))
- Fetch target info from Cargo even if `Build::target` is manually set ([#1299](https://github.com/rust-lang/cc-rs/pull/1299))
- Fix two files with different extensions having the same object name ([#1295](https://github.com/rust-lang/cc-rs/pull/1295))
- Allow disabling cc's ability to compile via env var CC_FORCE_DISABLE ([#1292](https://github.com/rust-lang/cc-rs/pull/1292))
- Regenerate target info ([#1293](https://github.com/rust-lang/cc-rs/pull/1293))

## [1.2.1](https://github.com/rust-lang/cc-rs/compare/cc-v1.2.0...cc-v1.2.1) - 2024-11-14

### Other

- When invoking `cl -?`, set stdin to null ([#1288](https://github.com/rust-lang/cc-rs/pull/1288))

## [1.2.0](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.37...cc-v1.2.0) - 2024-11-11

### Added

- add i686-pc-windows-gnullvm prefix detection ([#1283](https://github.com/rust-lang/cc-rs/pull/1283))

### Other

- Allow only specifying the architecture ([#1285](https://github.com/rust-lang/cc-rs/pull/1285))
- Fix WASM vs. WASI options ([#1284](https://github.com/rust-lang/cc-rs/pull/1284))

## [1.1.37](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.36...cc-v1.1.37) - 2024-11-08

### Other

- Use relative directory for obj files hash ([#1270](https://github.com/rust-lang/cc-rs/pull/1270))
- Regenerate target info ([#1280](https://github.com/rust-lang/cc-rs/pull/1280))

## [1.1.36](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.35...cc-v1.1.36) - 2024-11-05

### Other

- Fix CUDA build with clang++. ([#1273](https://github.com/rust-lang/cc-rs/pull/1273))

## [1.1.35](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.34...cc-v1.1.35) - 2024-11-04

### Other

- Remove support for FRC ([#1268](https://github.com/rust-lang/cc-rs/pull/1268))
- Do not add -fPIC by default on UEFI targets ([#1263](https://github.com/rust-lang/cc-rs/pull/1263))
- Use -windows-gnu for all UEFI targets ([#1264](https://github.com/rust-lang/cc-rs/pull/1264))

## [1.1.34](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.33...cc-v1.1.34) - 2024-11-02

### Other

- Remove redundant flags ([#1256](https://github.com/rust-lang/cc-rs/pull/1256))

## [1.1.33](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.32...cc-v1.1.33) - 2024-11-02

### Other

- Reduce size of `cc::Build`  and size of generated targets ([#1257](https://github.com/rust-lang/cc-rs/pull/1257))

## [1.1.32](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.31...cc-v1.1.32) - 2024-11-02

### Other

- Use `rustc`'s knowledge of LLVM/Clang target triples ([#1252](https://github.com/rust-lang/cc-rs/pull/1252))
- Use Cargo's target information when possible ([#1225](https://github.com/rust-lang/cc-rs/pull/1225))

## [1.1.31](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.30...cc-v1.1.31) - 2024-10-19

### Other

- Add comment explaining why cc does not rebuild on env PATH change ([#1247](https://github.com/rust-lang/cc-rs/pull/1247))

## [1.1.30](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.29...cc-v1.1.30) - 2024-10-11

### Other

- Don't pass -fPIC by default on wasm ([#1245](https://github.com/rust-lang/cc-rs/pull/1245))

## [1.1.29](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.28...cc-v1.1.29) - 2024-10-11

### Other

- Regenerate target info ([#1243](https://github.com/rust-lang/cc-rs/pull/1243))

## [1.1.28](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.27...cc-v1.1.28) - 2024-10-06

### Other

- Environment variables: For one accepting boolean, treat "0", "false" and empty env as false ([#1238](https://github.com/rust-lang/cc-rs/pull/1238))

## [1.1.27](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.26...cc-v1.1.27) - 2024-10-06

### Other

- Revert "Use debug version of MSVC runtime library on debug ([#1231](https://github.com/rust-lang/cc-rs/pull/1231))" ([#1237](https://github.com/rust-lang/cc-rs/pull/1237))
- Disable `CC_ENABLE_DEBUG_OUTPUT` if it is set to "0" ([#1234](https://github.com/rust-lang/cc-rs/pull/1234))

## [1.1.26](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.25...cc-v1.1.26) - 2024-10-06

### Other

- Use debug version of MSVC runtime library on debug ([#1231](https://github.com/rust-lang/cc-rs/pull/1231))

## [1.1.25](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.24...cc-v1.1.25) - 2024-10-05

### Other

- Remove incorrect "lib" prefixes in CXXSTDLIB doc comments ([#1228](https://github.com/rust-lang/cc-rs/pull/1228))

## [1.1.24](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.23...cc-v1.1.24) - 2024-10-01

### Other

- Fix wasm32-wasip1-threads:  shared-memory disallowed due to not compiled with 'atomics' or 'bulk-memory' features ([#1221](https://github.com/rust-lang/cc-rs/pull/1221))
- Reduce the need for the host target triple ([#1224](https://github.com/rust-lang/cc-rs/pull/1224))
- Add auto cancellation for CI jobs ([#1222](https://github.com/rust-lang/cc-rs/pull/1222))

## [1.1.23](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.22...cc-v1.1.23) - 2024-09-30

### Other

- Update doc for detecting changes/upgrades of compilers ([#1218](https://github.com/rust-lang/cc-rs/pull/1218))

## [1.1.22](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.21...cc-v1.1.22) - 2024-09-27

### Other

- Don't rerun if PATH changes ([#1215](https://github.com/rust-lang/cc-rs/pull/1215))

## [1.1.21](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.20...cc-v1.1.21) - 2024-09-18

### Other

- disable pic for targets that end in `-none` ([#1212](https://github.com/rust-lang/cc-rs/pull/1212))

## [1.1.20](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.19...cc-v1.1.20) - 2024-09-17

### Other

- Add buildcache as known Rust and C/C++ compiler wrapper ([#1209](https://github.com/rust-lang/cc-rs/pull/1209))

## [1.1.19](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.18...cc-v1.1.19) - 2024-09-15

### Other

- Add support arm64e-apple-darwin ([#1207](https://github.com/rust-lang/cc-rs/pull/1207))

## [1.1.18](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.17...cc-v1.1.18) - 2024-09-07

### Other
- Fixed unsoundness in `StderrForwarder::forward_available` ([#1203](https://github.com/rust-lang/cc-rs/pull/1203))

## [1.1.17](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.16...cc-v1.1.17) - 2024-09-06

### Fixed
- fix finding toolchains when invoked by msbuild ([#1201](https://github.com/rust-lang/cc-rs/pull/1201))

## [1.1.16](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.15...cc-v1.1.16) - 2024-09-04

### Other
- Treat VxWorks wr-cc as a Gnu compiler ([#1198](https://github.com/rust-lang/cc-rs/pull/1198))

## [1.1.15](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.14...cc-v1.1.15) - 2024-08-26

### Other
- Add -mfloat-abi=hard as a default argument when using any arm/thumb-none-eabihf target ([#1194](https://github.com/rust-lang/cc-rs/pull/1194))

## [1.1.14](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.13...cc-v1.1.14) - 2024-08-23

### Other
- allow finding tools from path if VisualStudioDir is set

## [1.1.13](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.12...cc-v1.1.13) - 2024-08-16

### Other
- Fix detect family: should detect emscripten as clang, closes [#1185](https://github.com/rust-lang/cc-rs/pull/1185) ([#1186](https://github.com/rust-lang/cc-rs/pull/1186))

## [1.1.12](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.11...cc-v1.1.12) - 2024-08-15

### Other
- improve docs ([#1183](https://github.com/rust-lang/cc-rs/pull/1183))

## [1.1.11](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.10...cc-v1.1.11) - 2024-08-14

### Other
- Add support for parsing shell encoded `*FLAGS` ([#1181](https://github.com/rust-lang/cc-rs/pull/1181))
- Replace vector of tuples with BTreeMap which already is sorted and free of duplicates ([#1177](https://github.com/rust-lang/cc-rs/pull/1177))

## [1.1.10](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.9...cc-v1.1.10) - 2024-08-11

### Other
- Remap Windows targets triples to their LLVM counterparts ([#1176](https://github.com/rust-lang/cc-rs/pull/1176))

## [1.1.9](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.8...cc-v1.1.9) - 2024-08-11

### Other
- Add custom CC wrapper to the wrapper whitelist ([#1175](https://github.com/rust-lang/cc-rs/pull/1175))

## [1.1.8](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.7...cc-v1.1.8) - 2024-08-06

### Other
- Fix broken link in docs.rs ([#1173](https://github.com/rust-lang/cc-rs/pull/1173))

## [1.1.7](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.6...cc-v1.1.7) - 2024-07-29

### Other
- add `.objects` ([#1166](https://github.com/rust-lang/cc-rs/pull/1166))

## [1.1.6](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.5...cc-v1.1.6) - 2024-07-19

### Other
- Clippy fixes ([#1163](https://github.com/rust-lang/cc-rs/pull/1163))

## [1.1.5](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.4...cc-v1.1.5) - 2024-07-15

### Other
- Fix cyclic compilation: Use vendored once_cell ([#1154](https://github.com/rust-lang/cc-rs/pull/1154))

## [1.1.4](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.3...cc-v1.1.4) - 2024-07-14

### Other
- Support compiling on wasm targets (Supersede [#1068](https://github.com/rust-lang/cc-rs/pull/1068)) ([#1160](https://github.com/rust-lang/cc-rs/pull/1160))

## [1.1.3](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.2...cc-v1.1.3) - 2024-07-14

### Other
- Reduce msrv to 1.63 ([#1158](https://github.com/rust-lang/cc-rs/pull/1158))
- Revert "Use raw-dylib for windows-sys ([#1137](https://github.com/rust-lang/cc-rs/pull/1137))" ([#1157](https://github.com/rust-lang/cc-rs/pull/1157))
- Fix typos ([#1152](https://github.com/rust-lang/cc-rs/pull/1152))
- Fix `doc_lazy_continuation` lints ([#1153](https://github.com/rust-lang/cc-rs/pull/1153))

## [1.1.2](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.1...cc-v1.1.2) - 2024-07-12

### Other
- Add empty `jobserver` feature. ([#1150](https://github.com/rust-lang/cc-rs/pull/1150))

## [1.1.1](https://github.com/rust-lang/cc-rs/compare/cc-v1.1.0...cc-v1.1.1) - 2024-07-12

### Other
- Fix is_flag_supported not respecting emit_rerun_if_env_changed ([#1147](https://github.com/rust-lang/cc-rs/pull/1147)) ([#1148](https://github.com/rust-lang/cc-rs/pull/1148))

## [1.1.0](https://github.com/rust-lang/cc-rs/compare/cc-v1.0.106...cc-v1.1.0) - 2024-07-08

### Added
- add cargo_output to eliminate last vestiges of stdout pollution ([#1141](https://github.com/rust-lang/cc-rs/pull/1141))

## [1.0.106](https://github.com/rust-lang/cc-rs/compare/cc-v1.0.105...cc-v1.0.106) - 2024-07-08

### Other
- Drop support for Visual Studio 12 (2013) ([#1046](https://github.com/rust-lang/cc-rs/pull/1046))
- Use raw-dylib for windows-sys ([#1137](https://github.com/rust-lang/cc-rs/pull/1137))
- Bump msrv to 1.67 ([#1143](https://github.com/rust-lang/cc-rs/pull/1143))
- Bump msrv to 1.65 ([#1140](https://github.com/rust-lang/cc-rs/pull/1140))
- Fix clippy warnings ([#1138](https://github.com/rust-lang/cc-rs/pull/1138))

## [1.0.105](https://github.com/rust-lang/cc-rs/compare/cc-v1.0.104...cc-v1.0.105) - 2024-07-07

### Other
- Regenerate windows sys bindings ([#1132](https://github.com/rust-lang/cc-rs/pull/1132))
- Fix generate-windows-sys-bindings ([#1133](https://github.com/rust-lang/cc-rs/pull/1133))
- Fix gen-windows-sys-binding ([#1130](https://github.com/rust-lang/cc-rs/pull/1130))
- Fix gen-windows-sys-binding ([#1127](https://github.com/rust-lang/cc-rs/pull/1127))
- Update windows-bindgen requirement from 0.57 to 0.58 ([#1123](https://github.com/rust-lang/cc-rs/pull/1123))

## [1.0.104](https://github.com/rust-lang/cc-rs/compare/cc-v1.0.103...cc-v1.0.104) - 2024-07-01

### Other
- Fixed link break about compile-time-requirements ([#1118](https://github.com/rust-lang/cc-rs/pull/1118))

## [1.0.103](https://github.com/rust-lang/cc-rs/compare/cc-v1.0.102...cc-v1.0.103) - 2024-06-30

### Other
- Fix compilation for wasm: env WASI_SYSROOT should be optional ([#1114](https://github.com/rust-lang/cc-rs/pull/1114))

## [1.0.102](https://github.com/rust-lang/cc-rs/compare/cc-v1.0.101...cc-v1.0.102) - 2024-06-29

### Other
- Fix invalid wasi targets compatibility ([#1105](https://github.com/rust-lang/cc-rs/pull/1105))
- Speedup regenerate-target-info and regenerate-windows-sys ([#1110](https://github.com/rust-lang/cc-rs/pull/1110))

## [1.0.101](https://github.com/rust-lang/cc-rs/compare/cc-v1.0.100...cc-v1.0.101) - 2024-06-25

### Other
- Use `Build::getenv` instead of `env::var*` in anywhere that makes sense ([#1103](https://github.com/rust-lang/cc-rs/pull/1103))

## [1.0.100](https://github.com/rust-lang/cc-rs/compare/cc-v1.0.99...cc-v1.0.100) - 2024-06-23

### Other
- Update publish.yml to use release-plz ([#1101](https://github.com/rust-lang/cc-rs/pull/1101))
- Accept `OsStr` instead of `str` for flags ([#1100](https://github.com/rust-lang/cc-rs/pull/1100))
- Use `dep:` syntax to avoid implicit features. ([#1099](https://github.com/rust-lang/cc-rs/pull/1099))
- Minor clippy fixes. ([#1098](https://github.com/rust-lang/cc-rs/pull/1098))
- Fix WASI compilation for C++ ([#1083](https://github.com/rust-lang/cc-rs/pull/1083))
- Regenerate windows sys bindings ([#1096](https://github.com/rust-lang/cc-rs/pull/1096))
- Rename regenerate-windows-sys to regenerate-windows-sys.yml ([#1095](https://github.com/rust-lang/cc-rs/pull/1095))
- Create regenerate-windows-sys.yml ([#1094](https://github.com/rust-lang/cc-rs/pull/1094))
- Update windows-bindgen requirement from 0.56 to 0.57 ([#1091](https://github.com/rust-lang/cc-rs/pull/1091))
- Eagerly close tempfile to fix [#1082](https://github.com/rust-lang/cc-rs/pull/1082) ([#1087](https://github.com/rust-lang/cc-rs/pull/1087))
- Output msvc.exe in the output directory ([#1090](https://github.com/rust-lang/cc-rs/pull/1090))
- Fix clippy warnings on Windows ([#1088](https://github.com/rust-lang/cc-rs/pull/1088))
- Don't try to free DLL on drop ([#1089](https://github.com/rust-lang/cc-rs/pull/1089))
- Fix panic safety issue in StderrForwarder ([#1079](https://github.com/rust-lang/cc-rs/pull/1079))
