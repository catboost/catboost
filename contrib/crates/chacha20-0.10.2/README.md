# RustCrypto: ChaCha20

[![Crate][crate-image]][crate-link]
[![Docs][docs-image]][docs-link]
[![Build Status][build-image]][build-link]
![Apache2/MIT licensed][license-image]
![Rust Version][rustc-image]
[![Project Chat][chat-image]][chat-link]
[![HAZMAT][hazmat-image]][hazmat-link]

Implementation of the [ChaCha] family of stream ciphers.

ChaCha improves upon the previous [Salsa] family of stream ciphers
with increased per-round diffusion at no cost to performance.

This crate also contains an implementation of the [XChaCha] family of stream ciphers
with an extended 192-bit (24-byte) nonce, gated under the `xchacha` Cargo feature,
and "legacy" (a.k.a "djb") variant with 64-bit nonce, gated under the `legacy` crate feature.

**WARNING:** This implementation internally uses 32-bit counter,
while the original "legacy" variant implementation uses 64-bit counter.
In other words, it does not allow encryption of more than 256 GiB of data.

## Security

### ⚠️ Warning: [Hazmat!][hazmat-link]

This crate does not ensure ciphertexts are authentic (i.e. by using a MAC to
verify ciphertext integrity), which can lead to serious vulnerabilities
if used incorrectly!

To avoid this, use an [AEAD] mode based on ChaCha20, e.g. [`chacha20poly1305`].
See the [RustCrypto/AEADs] repository for more information.

USE AT YOUR OWN RISK!

### Notes

This crate has received one [security audit by NCC Group][NCC-AUDIT], with no significant
findings. We would like to thank [MobileCoin] for funding the audit.

All implementations contained in the crate (along with the underlying ChaCha20
stream cipher itself) are designed to execute in constant time.

## Examples

```rust
// This example requires `cipher` crate feature
#[cfg(feature = "cipher")] {

use chacha20::ChaCha20;
use chacha20::cipher::{KeyIvInit, StreamCipher, StreamCipherSeek};
use hex_literal::hex;

let key = [0x42; 32];
let nonce = [0x24; 12];
let plaintext = hex!("000102030405060708090A0B0C0D0E0F");
let ciphertext = hex!("e405626e4f1236b3670ee428332ea20e");

// Key and IV must be references to the `Array` type.
// Here we use the `Into` trait to convert arrays into it.
let mut cipher = ChaCha20::new(&key.into(), &nonce.into());

let mut buffer = plaintext;

// apply keystream (encrypt)
cipher.apply_keystream(&mut buffer);
assert_eq!(buffer, ciphertext);

let ciphertext = buffer;

// ChaCha ciphers support seeking
cipher.seek(0u32);

// decrypt ciphertext by applying keystream again
cipher.apply_keystream(&mut buffer);
assert_eq!(buffer, plaintext);

// stream ciphers can be used with streaming messages
cipher.seek(0u32);
for chunk in buffer.chunks_mut(3) {
    cipher.apply_keystream(chunk);
}
assert_eq!(buffer, ciphertext);
}
```

## Configuration Flags

You can modify crate using the following configuration flags:

- `chacha20_backend="avx2"`: force AVX2 backend on x86/x86_64 targets.
  Requires enabled AVX2 target feature. Ignored on non-x86(_64) targets.
- `chacha20_backend="avx512"`: force AVX-512 backend on x86/x86_64 targets.
  Requires enabled AVX-512 target feature (MSRV 1.89). Ignored on non-x86(_64) targets.
- `chacha20_backend="soft"`: force software backend.
- `chacha20_backend="sse2"`: force SSE2 backend on x86/x86_64 targets.
  Requires enabled SSE2 target feature. Ignored on non-x86(-64) targets.

To use the MSRV 1.89 AVX-512 support with autodetection, you must enable it using
`chacha20_avx512` configuration flag.

The flags can be enabled using `RUSTFLAGS` environmental variable
(e.g. `RUSTFLAGS='--cfg chacha20_backend="avx2"'`) or by modifying `.cargo/config.toml`.

## License

Licensed under either of:

- [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0)
- [MIT license](http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

[//]: # (badges)

[crate-image]: https://img.shields.io/crates/v/chacha20.svg
[crate-link]: https://crates.io/crates/chacha20
[docs-image]: https://docs.rs/chacha20/badge.svg
[docs-link]: https://docs.rs/chacha20/
[build-image]: https://github.com/RustCrypto/stream-ciphers/actions/workflows/chacha20.yml/badge.svg
[build-link]: https://github.com/RustCrypto/stream-ciphers/actions/workflows/chacha20.yml
[license-image]: https://img.shields.io/badge/license-Apache2.0/MIT-blue.svg
[rustc-image]: https://img.shields.io/badge/rustc-1.85+-blue.svg
[chat-image]: https://img.shields.io/badge/zulip-join_chat-blue.svg
[chat-link]: https://rustcrypto.zulipchat.com/#narrow/stream/260049-stream-ciphers
[hazmat-image]: https://img.shields.io/badge/crypto-hazmat%E2%9A%A0-red.svg
[hazmat-link]: https://github.com/RustCrypto/meta/blob/master/HAZMAT.md

[//]: # (footnotes)

[ChaCha]: https://en.wikipedia.org/wiki/Salsa20#ChaCha_variant
[Salsa]: https://en.wikipedia.org/wiki/Salsa20
[XChaCha]: https://tools.ietf.org/html/draft-arciszewski-xchacha-02
[AEAD]: https://en.wikipedia.org/wiki/Authenticated_encryption
[`chacha20poly1305`]: https://docs.rs/chacha20poly1305
[RustCrypto/AEADs]: https://github.com/RustCrypto/AEADs
[NCC-AUDIT]: https://web.archive.org/web/20240108154854/https://research.nccgroup.com/wp-content/uploads/2020/02/NCC_Group_MobileCoin_RustCrypto_AESGCM_ChaCha20Poly1305_Implementation_Review_2020-02-12_v1.0.pdf
[MobileCoin]: https://www.mobilecoin.com/
