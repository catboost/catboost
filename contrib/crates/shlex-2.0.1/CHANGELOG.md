# 2.0.1

* Fixes a compile error when building the documentation.

# 2.0.0

* **Breaking:** Items that were marked as deprecated in 1.x have been removed: `join`, `quote`, `bytes::join`, and `bytes::quote`.
* **Breaking:** The `DerefMut` impl for `Shlex` has been removed since it was unsound. New `unsafe` APIs have been added in its place: `Shlex::from_bytes`, `Shlex::as_bytes_mut`.

# 1.3.0

* Full fix for the high-severity security vulnerability [RUSTSEC-2024-0006](https://rustsec.org/advisories/RUSTSEC-2024-0006.html) a.k.a. [GHSA-r7qv-8r2h-pg27](https://github.com/comex/rust-shlex/security/advisories/GHSA-r7qv-8r2h-pg27):
    * Deprecates quote APIs in favor of `try_` equivalents that complain about nul bytes.
    * Also adds a builder API, which allows re-enabling nul bytes without using the deprecated interface, and in the future can allow other things (as discussed in quoting_warning).
    * Adds documentation about various security risks that remain, particularly with interactive shells.
* Adds explicit MSRV of 1.46.0.

# 1.2.1

* Partial fix for the high-severity security vulnerability [RUSTSEC-2024-0006](https://rustsec.org/advisories/RUSTSEC-2024-0006.html) a.k.a. [GHSA-r7qv-8r2h-pg27](https://github.com/comex/rust-shlex/security/advisories/GHSA-r7qv-8r2h-pg27) without bumping MSRV:
    * The bytes `{` and `\xa0` are now escaped by quoting functions.

# 1.2.0

* Adds `bytes` module to support operating directly on byte strings.

# 1.1.0

* Adds the `std` feature (enabled by default).
* Disabling the `std` feature makes the crate work in `#![no_std]` mode, assuming presence of the `alloc` crate.

# 1.0.0

* Adds the `join` convenience function.
* Fixes parsing of `'\\n'` to match the behavior of bash/Zsh/Python `shlex`. The result was previously `\n`, now it is `\\n`.

# 0.1.1

* Adds handling of `#` comments.

# 0.1.0

This is the initial release.
