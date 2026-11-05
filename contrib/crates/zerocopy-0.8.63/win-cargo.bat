@rem Copyright 2024 The Fuchsia Authors

@rem Licensed under a BSD-style license <LICENSE-BSD>, Apache License, Version 2.0
@rem <LICENSE-APACHE or https://www.apache.org/licenses/LICENSE-2.0>, or the MIT
@rem license <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your option.
@rem This file may not be copied, modified, or distributed except according to
@rem those terms.

@rem Build `cargo-zerocopy` without any RUSTFLAGS set in the environment.
@rem Build from the repository root so that Zerocopy's vendoring config does
@rem not apply to the unvendored tools workspace.
@set SCRIPT_DIR=%~dp0
@set TEMP_RUSTFLAGS=%RUSTFLAGS%
@set RUSTFLAGS=
@pushd "%SCRIPT_DIR%.."
@cargo +stable build --manifest-path tools\Cargo.toml -p cargo-zerocopy -q
@set CARGO_ZEROCOPY_BUILD_STATUS=%ERRORLEVEL%
@popd
@set RUSTFLAGS=%TEMP_RUSTFLAGS%
@set TEMP_RUSTFLAGS=
@if not "%CARGO_ZEROCOPY_BUILD_STATUS%"=="0" exit /b %CARGO_ZEROCOPY_BUILD_STATUS%
@rem Thin wrapper around the `cargo-zerocopy` binary in `tools/cargo-zerocopy`
@pushd "%SCRIPT_DIR%"
@..\tools\target\debug\cargo-zerocopy %*
@set CARGO_ZEROCOPY_STATUS=%ERRORLEVEL%
@popd
@exit /b %CARGO_ZEROCOPY_STATUS%
