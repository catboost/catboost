#![no_std]

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(not(miri))]
#[inline]
#[must_use]
fn optimizer_hide(mut value: u8) -> u8 {
    // SAFETY: the input value is passed unchanged to the output, the inline assembly does nothing.
    unsafe {
        core::arch::asm!("/* {0} */", inout(reg_byte) value, options(pure, nomem, nostack, preserves_flags));
        value
    }
}

#[cfg(any(
    target_arch = "arm",
    target_arch = "aarch64",
    target_arch = "riscv32",
    target_arch = "riscv64"
))]
#[cfg(not(miri))]
#[inline]
#[must_use]
#[allow(asm_sub_register)]
fn optimizer_hide(mut value: u8) -> u8 {
    // SAFETY: the input value is passed unchanged to the output, the inline assembly does nothing.
    unsafe {
        core::arch::asm!("/* {0} */", inout(reg) value, options(pure, nomem, nostack, preserves_flags));
        value
    }
}

#[cfg(any(
    not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "arm",
        target_arch = "aarch64",
        target_arch = "riscv32",
        target_arch = "riscv64",
    )),
    miri,
))]
#[inline(never)]
#[must_use]
fn optimizer_hide(value: u8) -> u8 {
    // The current implementation of black_box in the main codegen backends is similar to
    // {
    //     let result = value;
    //     asm!("", in(reg) &result);
    //     result
    // }
    // which round-trips the value through the stack, instead of leaving it in a register.
    // Experimental codegen backends might implement black_box as a pure identity function,
    // without the expected optimization barrier, so it's less guaranteed than inline asm.
    // For that reason, we also use the #[inline(never)] hint, which makes it harder for an
    // optimizer to look inside this function.
    core::hint::black_box(value)
}

#[inline]
#[must_use]
fn constant_time_ne(a: &[u8], b: &[u8]) -> u8 {
    assert!(a.len() == b.len());

    // These useless slices make the optimizer elide the bounds checks.
    // See the comment in clone_from_slice() added on Rust commit 6a7bc47.
    let len = a.len();
    let a = &a[..len];
    let b = &b[..len];

    let mut tmp = 0;
    for i in 0..len {
        tmp |= a[i] ^ b[i];
    }

    // The compare with 0 must happen outside this function.
    optimizer_hide(tmp)
}

/// Compares two equal-sized byte strings in constant time.
///
/// # Examples
///
/// ```
/// use constant_time_eq::constant_time_eq;
///
/// assert!(constant_time_eq(b"foo", b"foo"));
/// assert!(!constant_time_eq(b"foo", b"bar"));
/// assert!(!constant_time_eq(b"bar", b"baz"));
/// # assert!(constant_time_eq(b"", b""));
///
/// // Not equal-sized, so won't take constant time.
/// assert!(!constant_time_eq(b"foo", b""));
/// assert!(!constant_time_eq(b"foo", b"quux"));
/// ```
#[must_use]
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && constant_time_ne(a, b) == 0
}

// Fixed-size array variant.

#[inline]
#[must_use]
fn constant_time_ne_n<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u8 {
    let mut tmp = 0;
    for i in 0..N {
        tmp |= a[i] ^ b[i];
    }

    // The compare with 0 must happen outside this function.
    optimizer_hide(tmp)
}

/// Compares two fixed-size byte strings in constant time.
///
/// # Examples
///
/// ```
/// use constant_time_eq::constant_time_eq_n;
///
/// assert!(constant_time_eq_n(&[3; 20], &[3; 20]));
/// assert!(!constant_time_eq_n(&[3; 20], &[7; 20]));
/// ```
#[must_use]
pub fn constant_time_eq_n<const N: usize>(a: &[u8; N], b: &[u8; N]) -> bool {
    constant_time_ne_n(a, b) == 0
}

// Fixed-size variants for the most common sizes.

/// Compares two 128-bit byte strings in constant time.
///
/// # Examples
///
/// ```
/// use constant_time_eq::constant_time_eq_16;
///
/// assert!(constant_time_eq_16(&[3; 16], &[3; 16]));
/// assert!(!constant_time_eq_16(&[3; 16], &[7; 16]));
/// ```
#[inline]
#[must_use]
pub fn constant_time_eq_16(a: &[u8; 16], b: &[u8; 16]) -> bool {
    constant_time_eq_n(a, b)
}

/// Compares two 256-bit byte strings in constant time.
///
/// # Examples
///
/// ```
/// use constant_time_eq::constant_time_eq_32;
///
/// assert!(constant_time_eq_32(&[3; 32], &[3; 32]));
/// assert!(!constant_time_eq_32(&[3; 32], &[7; 32]));
/// ```
#[inline]
#[must_use]
pub fn constant_time_eq_32(a: &[u8; 32], b: &[u8; 32]) -> bool {
    constant_time_eq_n(a, b)
}

/// Compares two 512-bit byte strings in constant time.
///
/// # Examples
///
/// ```
/// use constant_time_eq::constant_time_eq_64;
///
/// assert!(constant_time_eq_64(&[3; 64], &[3; 64]));
/// assert!(!constant_time_eq_64(&[3; 64], &[7; 64]));
/// ```
#[inline]
#[must_use]
pub fn constant_time_eq_64(a: &[u8; 64], b: &[u8; 64]) -> bool {
    constant_time_eq_n(a, b)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "count_instructions_test")]
    extern crate std;

    #[cfg(feature = "count_instructions_test")]
    #[test]
    fn count_optimizer_hide_instructions() -> std::io::Result<()> {
        use super::optimizer_hide;
        use count_instructions::count_instructions;

        fn count() -> std::io::Result<usize> {
            // If optimizer_hide does not work, constant propagation and folding
            // will make this identical to count_optimized() below.
            let mut count = 0;
            assert_eq!(
                10u8,
                count_instructions(
                    || optimizer_hide(1)
                        + optimizer_hide(2)
                        + optimizer_hide(3)
                        + optimizer_hide(4),
                    |_| count += 1
                )?
            );
            Ok(count)
        }

        fn count_optimized() -> std::io::Result<usize> {
            #[inline]
            fn inline_identity(value: u8) -> u8 {
                value
            }

            let mut count = 0;
            assert_eq!(
                10u8,
                count_instructions(
                    || inline_identity(1)
                        + inline_identity(2)
                        + inline_identity(3)
                        + inline_identity(4),
                    |_| count += 1
                )?
            );
            Ok(count)
        }

        assert!(count()? > count_optimized()?);
        Ok(())
    }
}
