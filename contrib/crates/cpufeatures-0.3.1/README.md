# [RustCrypto]: CPU Feature Detection

[![crate][crate-image]][crate-link]
[![Docs][docs-image]][docs-link]
[![Build Status][build-image]][build-link]
![Apache2/MIT licensed][license-image]
![Rust Version][rustc-image]
[![Project Chat][chat-image]][chat-link]

Lightweight and efficient runtime CPU feature detection for `aarch64`, `loongarch64`, and
`x86`/`x86_64` targets.

Supports `no_std` as well as mobile targets including iOS and Android,
providing an alternative to the `std`-dependent `is_x86_feature_detected!`
macro.

This crate is intended as a stopgap until Rust [RFC 2725] adding first-class
target feature detection macros to `libcore` is implemented.

## Example
```rust
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86_backend {
    // This macro creates `cpuid_aes_sha` module
    cpufeatures::new!(cpuid_aes_sha, "aes", "sha");

    pub fn run() {
        // `token` is a Zero Sized Type (ZST) value, which guarantees
        // that underlying static storage got properly initialized,
        // which allows to omit initialization branch
        let token: cpuid_aes_sha::InitToken = cpuid_aes_sha::init();

        if token.get() {
            println!("CPU supports both SHA and AES extensions");
        } else {
            println!("SHA and AES extensions are not supported");
        }

        // If stored value needed only once you can get stored value
        // omitting the token
        let val = cpuid_aes_sha::get();
        assert_eq!(val, token.get());
    
        // Additionally you can get both token and value
        let (token, val) = cpuid_aes_sha::init_get();
        assert_eq!(val, token.get());
    }
}
```

Note that if all tested target features are enabled via compiler options
(e.g. by using `RUSTFLAGS`), the `get` method will always return `true`
and `init` will not use CPUID instruction. Such behavior allows
compiler to completely eliminate fallback code.

After first call macro caches result and returns it in subsequent
calls, thus runtime overhead for them is minimal.

## Supported target architectures

*NOTE: target features with an asterisk are unstable (nightly-only) and subject
to change to match upstream name changes in the Rust standard library.*

### `aarch64`

Linux, iOS, and macOS/ARM only (ARM64 does not support OS-independent feature detection)

Target features:

- `aes`*
- `sha2`*
- `sha3`*

### `loongarch64`

Linux only (`loongarch64` does not support OS-independent feature detection)

Target features:

- `32s`*
- `f`*
- `d`*
- `frecipe`*
- `div32`*
- `lsx`*
- `lasx`*
- `lam-bh`*
- `lamcas`*
- `ld-seq-sa`*
- `scq`*
- `lbt`*
- `lvz`*
- `ual`*

### `x86`/`x86_64`

OS independent and `no_std`-friendly

Target features:

- `adx`
- `aes`
- `avx`
- `avx2`
- `avx512bw`*
- `avx512cd`*
- `avx512dq`*
- `avx512er`*
- `avx512f`*
- `avx512ifma`*
- `avx512pf`*
- `avx512vl`*
- `avx512vbmi`*
- `avx512vbmi2`*
- `bmi1`
- `bmi2`
- `fma`
- `mmx`
- `pclmulqdq`
- `popcnt`
- `rdrand`
- `rdseed`
- `sgx`
- `sha`
- `sse`
- `sse2`
- `sse3`
- `sse4.1`
- `sse4.2`
- `ssse3`
- `sha512`
- `sm3`
- `sm4`

If you would like detection support for a target feature which is not on
this list, please [open a GitHub issue][github-issue].

## License

Licensed under either of:

 * [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0)
 * [MIT license](http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

[//]: # (badges)

[crate-image]: https://img.shields.io/crates/v/cpufeatures.svg
[crate-link]: https://crates.io/crates/cpufeatures
[docs-image]: https://docs.rs/cpufeatures/badge.svg
[docs-link]: https://docs.rs/cpufeatures/
[license-image]: https://img.shields.io/badge/license-Apache2.0/MIT-blue.svg
[rustc-image]: https://img.shields.io/badge/rustc-1.85+-blue.svg
[chat-image]: https://img.shields.io/badge/zulip-join_chat-blue.svg
[chat-link]: https://rustcrypto.zulipchat.com/#narrow/stream/260052-utils
[build-image]: https://github.com/RustCrypto/utils/actions/workflows/cpufeatures.yml/badge.svg?branch=master
[build-link]: https://github.com/RustCrypto/utils/actions/workflows/cpufeatures.yml?query=branch:master

[//]: # (general links)

[RustCrypto]: https://github.com/RustCrypto
[RustCrypto/utils#378]: https://github.com/RustCrypto/utils/issues/378
[RFC 2725]: https://github.com/rust-lang/rfcs/pull/2725
[github-issue]: https://github.com/RustCrypto/utils/issues/new?title=cpufeatures:%20requesting%20support%20for%20CHANGEME%20target%20feature
