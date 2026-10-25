pub fn slide_hash(state: &mut crate::deflate::State) {
    let wsize = state.w_size as u16;

    // The state.head and state.prev slices have a length that is a power of 2 between 8 and 16.
    // That knowledge means `chunks_exact` with a (small) power of 2 can be used without risk of
    // missing elements.
    slide_hash_chain(state.head.as_mut_slice(), wsize);
    slide_hash_chain(state.prev.as_mut_slice(), wsize);
}

fn slide_hash_chain(table: &mut [u16], wsize: u16) {
    #[cfg(target_arch = "x86_64")]
    if crate::cpu_features::is_enabled_avx2_and_bmi2() {
        // SAFETY: the avx2 and bmi2 target feature are enabled.
        return unsafe { avx2::slide_hash_chain(table, wsize) };
    }

    #[cfg(target_arch = "aarch64")]
    if crate::cpu_features::is_enabled_neon() {
        return unsafe { neon::slide_hash_chain(table, wsize) };
    }

    // FIXME: Remove lsx feature guard once MSRV is 1.89.0 or higher
    #[cfg(all(target_arch = "loongarch64", feature = "lsx"))]
    if crate::cpu_features::is_enabled_lsx() {
        return unsafe { lsx::slide_hash_chain(table, wsize) };
    }

    #[cfg(target_arch = "wasm32")]
    if crate::cpu_features::is_enabled_simd128() {
        // SAFETY: the simd128 target feature is enabled.
        return unsafe { wasm::slide_hash_chain(table, wsize) };
    }

    rust::slide_hash_chain(table, wsize);
}

#[inline(always)]
fn generic_slide_hash_chain<const N: usize>(table: &mut [u16], wsize: u16) {
    debug_assert_eq!(table.len() % N, 0);

    for chunk in table.chunks_exact_mut(N) {
        for m in chunk.iter_mut() {
            *m = m.saturating_sub(wsize);
        }
    }
}

mod rust {
    pub fn slide_hash_chain(table: &mut [u16], wsize: u16) {
        // 32 means that 4 128-bit values can be processed per iteration. That appear to be the
        // optimal amount on x86_64 (SSE) and aarch64 (NEON).
        super::generic_slide_hash_chain::<32>(table, wsize);
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    /// # Safety
    ///
    /// Behavior is undefined if the `avx2` target feature is not enabled
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "bmi2")]
    #[target_feature(enable = "bmi1")]
    pub unsafe fn slide_hash_chain(table: &mut [u16], wsize: u16) {
        // 64 means that 4 256-bit values can be processed per iteration.
        // That appear to be the optimal amount for avx2.
        //
        // This vectorizes well https://godbolt.org/z/sGbdYba7K
        super::generic_slide_hash_chain::<64>(table, wsize);
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    /// # Safety
    ///
    /// Behavior is undefined if the `neon` target feature is not enabled
    #[target_feature(enable = "neon")]
    pub unsafe fn slide_hash_chain(table: &mut [u16], wsize: u16) {
        // 32 means that 4 128-bit values can be processed per iteration. That appear to be the
        // optimal amount for neon.
        super::generic_slide_hash_chain::<32>(table, wsize);
    }
}

// FIXME: Remove lsx feature guard once MSRV is 1.89.0 or higher
#[cfg(all(target_arch = "loongarch64", feature = "lsx"))]
mod lsx {
    /// # Safety
    ///
    /// Behavior is undefined if the `lsx` target feature is not enabled.
    #[target_feature(enable = "lsx")]
    pub unsafe fn slide_hash_chain(table: &mut [u16], wsize: u16) {
        // 32 means that 4 128-bit values can be processed per iteration. That appear to be the
        // optimal amount for LSX.
        super::generic_slide_hash_chain::<32>(table, wsize);
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    /// # Safety
    ///
    /// Behavior is undefined if the `simd128` target feature is not enabled
    #[target_feature(enable = "simd128")]
    pub unsafe fn slide_hash_chain(table: &mut [u16], wsize: u16) {
        // 32 means that 4 128-bit values can be processed per iteration. That appear to be the
        // optimal amount on x86_64 (SSE) and aarch64 (NEON), which is what this will ultimately
        // compile down to.
        super::generic_slide_hash_chain::<32>(table, wsize);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WSIZE: u16 = 32768;

    const INPUT: [u16; 64] = [
        0, 0, 28790, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 43884, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 64412, 0, 0, 0, 0, 0, 21043, 0, 0, 0, 0, 0, 23707, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 64026, 0, 0, 20182,
    ];

    const OUTPUT: [u16; 64] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11116, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 31644, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 31258, 0, 0, 0,
    ];

    #[test]
    fn test_slide_hash_rust() {
        let mut input = INPUT;

        rust::slide_hash_chain(&mut input, WSIZE);

        assert_eq!(input, OUTPUT);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_slide_hash_avx2() {
        if crate::cpu_features::is_enabled_avx2_and_bmi2() {
            let mut input = INPUT;

            unsafe { avx2::slide_hash_chain(&mut input, WSIZE) };

            assert_eq!(input, OUTPUT);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_slide_hash_neon() {
        if crate::cpu_features::is_enabled_neon() {
            let mut input = INPUT;

            unsafe { neon::slide_hash_chain(&mut input, WSIZE) };

            assert_eq!(input, OUTPUT);
        }
    }

    // FIXME: Remove lsx feature guard once MSRV is 1.89.0 or higher
    #[test]
    #[cfg(all(target_arch = "loongarch64", feature = "lsx"))]
    fn test_slide_hash_lsx() {
        if crate::cpu_features::is_enabled_lsx() {
            let mut input = INPUT;

            unsafe { lsx::slide_hash_chain(&mut input, WSIZE) };

            assert_eq!(input, OUTPUT);
        }
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_slide_hash_wasm() {
        if crate::cpu_features::is_enabled_simd128() {
            let mut input = INPUT;

            unsafe { wasm::slide_hash_chain(&mut input, WSIZE) };

            assert_eq!(input, OUTPUT);
        }
    }

    quickcheck::quickcheck! {
        fn slide_is_rust_slide(v: Vec<u16>, wsize: u16) -> bool {
            // pad to a multiple of 64 (the biggest chunk size currently in use)
            let difference = v.len().next_multiple_of(64) - v.len();
            let mut v = v;
            v.extend(core::iter::repeat(u16::MAX).take(difference));


            let mut a = v.clone();
            let mut b = v;

            rust::slide_hash_chain(&mut a, wsize);
            slide_hash_chain(&mut b, wsize);

            a == b
        }
    }
}
