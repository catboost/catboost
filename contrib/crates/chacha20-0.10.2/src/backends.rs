#![cfg(any(feature = "cipher", feature = "rng"))]

use cfg_if::cfg_if;

cfg_if! {
    if #[cfg(chacha20_backend = "soft")] {
        pub(crate) mod soft;
    } else if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
        cfg_if! {
            if #[cfg(all(chacha20_avx512, chacha20_backend = "avx512"))] {
                pub(crate) mod avx512;
                // AVX-2 backend needed for RNG if enabled
                #[cfg(feature = "rng")]
                pub(crate) mod avx2;
            } else if #[cfg(chacha20_backend = "avx2")] {
                pub(crate) mod avx2;
            } else if #[cfg(chacha20_backend = "sse2")] {
                pub(crate) mod sse2;
            } else {
                pub(crate) mod soft;
                #[cfg(chacha20_avx512)]
                pub(crate) mod avx512;
                pub(crate) mod avx2;
                pub(crate) mod sse2;
            }
        }
    } else if #[cfg(all(target_arch = "aarch64", target_feature = "neon"))] {
        pub(crate) mod neon;
    } else {
        pub(crate) mod soft;
    }
}
