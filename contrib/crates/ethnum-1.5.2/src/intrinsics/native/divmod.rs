//! This module contains a Rust port of the `__u?divmodti4` compiler builtins
//! that are typically used for implementing 64-bit signed and unsigned division
//! on 32-bit platforms.
//!
//! This port is adapted to use 128-bit high and low words in order to implement
//! 256-bit division.
//!
//! This source is ported from LLVM project from C:
//! - signed division: <https://github.com/llvm/llvm-project/blob/main/compiler-rt/lib/builtins/divmodti4.c>
//! - unsigned division: <https://github.com/llvm/llvm-project/blob/main/compiler-rt/lib/builtins/udivmodti4.c>

use crate::{int::I256, uint::U256};
use core::{mem::MaybeUninit, num::NonZeroU128};

#[inline(always)]
fn udiv256_by_128_to_128(u1: u128, u0: u128, mut v: NonZeroU128, r: &mut u128) -> u128 {
    const N_UDWORD_BITS: u32 = 128;

    #[inline]
    unsafe fn shl_nz(x: NonZeroU128, n: u32) -> NonZeroU128 {
        debug_assert!(n < N_UDWORD_BITS);
        let res: u128 = x.get() << n;
        debug_assert_ne!(res, 0);
        NonZeroU128::new_unchecked(res)
    }

    #[inline]
    unsafe fn shr_nz(x: NonZeroU128, n: u32) -> NonZeroU128 {
        debug_assert!(n < N_UDWORD_BITS);
        let res: u128 = x.get() >> n;
        debug_assert_ne!(res, 0);
        NonZeroU128::new_unchecked(res)
    }

    const B: u128 = 1 << (N_UDWORD_BITS / 2); // Number base (128 bits)
    let (un1, un0): (u128, u128); // Norm. dividend LSD's
    let (vn1, vn0): (NonZeroU128, u128); // Norm. divisor digits
    let (mut q1, mut q0): (u128, u128); // Quotient digits
    let (un128, un21, un10): (u128, u128, u128); // Dividend digit pairs

    debug_assert!(v.get() > u1);

    let s = v.leading_zeros();
    debug_assert_ne!(s, N_UDWORD_BITS);
    if s > 0 {
        // Normalize the divisor.
        v = unsafe { shl_nz(v, s) };
        un128 = (u1 << s) | (u0 >> (N_UDWORD_BITS - s));
        un10 = u0 << s; // Shift dividend left
    } else {
        // Avoid undefined behavior of (u0 >> 128).
        un128 = u1;
        un10 = u0;
    }

    // Break divisor up into two 64-bit digits.
    vn1 = unsafe { shr_nz(v, N_UDWORD_BITS / 2) };
    vn0 = v.get() & 0xFFFF_FFFF_FFFF_FFFF;

    // Break right half of dividend into two digits.
    un1 = un10 >> (N_UDWORD_BITS / 2);
    un0 = un10 & 0xFFFF_FFFF_FFFF_FFFF;

    // Compute the first quotient digit, q1.
    q1 = un128 / vn1;
    let mut rhat = un128 - q1 * vn1.get();

    // q1 has at most error 2. No more than 2 iterations.
    while q1 >= B || q1 * vn0 > B * rhat + un1 {
        q1 -= 1;
        rhat += vn1.get();
        if rhat >= B {
            break;
        }
    }

    un21 = un128
        .wrapping_mul(B)
        .wrapping_add(un1)
        .wrapping_sub(q1.wrapping_mul(v.get()));

    // Compute the second quotient digit.
    q0 = un21 / vn1;
    rhat = un21 - q0 * vn1.get();

    // q0 has at most error 2. No more than 2 iterations.
    while q0 >= B || q0 * vn0 > B * rhat + un0 {
        q0 -= 1;
        rhat += vn1.get();
        if rhat >= B {
            break;
        }
    }

    *r = (un21
        .wrapping_mul(B)
        .wrapping_add(un0)
        .wrapping_sub(q0.wrapping_mul(v.get())))
        >> s;
    q1 * B + q0
}

#[allow(clippy::many_single_char_names)]
pub fn udivmod4(
    res: &mut MaybeUninit<U256>,
    a: &U256,
    b: &U256,
    rem: Option<&mut MaybeUninit<U256>>,
) {
    // In the LLVM version on the x86_64 platform, `udiv256_by_128_to_128` would
    // defer to `divq` instruction, which divides a 128-bit value by a 64-bit
    // one returning a 64-bit value, making it very performant when dividing
    // small values:
    // ```
    //   du_int result;
    //   __asm__("divq %[v]"
    //           : "=a"(result), "=d"(*r)
    //           : [ v ] "r"(v), "a"(u0), "d"(u1));
    //   return result;
    // ```
    // Unfortunately, there is no 256-bit equivalent on x86_64, but we can still
    // shortcut if the high and low values of the operands are 0:
    if a.high() | b.high() == 0 {
        res.write(U256::from_words(0, a.low() / b.low()));
        if let Some(rem) = rem {
            rem.write(U256::from_words(0, a.low() % b.low()));
        }
        return;
    }

    let dividend = *a;
    let divisor = *b;
    let quotient: U256;
    let mut remainder: U256;

    if divisor > dividend {
        if let Some(rem) = rem {
            rem.write(dividend);
        }
        res.write(U256::ZERO);
        return;
    }
    // When the divisor fits in 128 bits, we can use an optimized path.
    if *divisor.high() == 0 {
        remainder = U256::ZERO;
        if dividend.high() < divisor.low() {
            // The result fits in 128 bits.
            quotient = U256::from_words(
                0,
                udiv256_by_128_to_128(
                    *dividend.high(),
                    *dividend.low(),
                    // SAFETY: dividend.high() < divisor.low()
                    unsafe { NonZeroU128::new_unchecked(*divisor.low()) },
                    remainder.low_mut(),
                ),
            );
        } else {
            // First, divide with the high part to get the remainder in dividend.s.high.
            // After that dividend.s.high < divisor.s.low.
            quotient = U256::from_words(
                dividend.high() / divisor.low(),
                udiv256_by_128_to_128(
                    dividend.high() % divisor.low(),
                    *dividend.low(),
                    // SAFETY: dividend.high() / divisor.low()
                    unsafe { NonZeroU128::new_unchecked(*divisor.low()) },
                    remainder.low_mut(),
                ),
            );
        }
        if let Some(rem) = rem {
            rem.write(remainder);
        }
        res.write(quotient);
        return;
    }

    // SAFETY: `*divisor.high() != 0`
    (quotient, remainder) = unsafe { div_mod_knuth(&dividend, &divisor) };

    if let Some(rem) = rem {
        rem.write(remainder);
    }
    res.write(quotient);
}

// See Knuth, TAOCP, Volume 2, section 4.3.1, Algorithm D.
// https://skanthak.homepage.t-online.de/division.html
// SAFETY: The high word of v (the divisor) must be non-zero.
#[inline]
unsafe fn div_mod_knuth(u: &U256, v: &U256) -> (U256, U256) {
    const N_UDWORD_BITS: u32 = 128;
    debug_assert_ne!(
        *u.high(),
        0,
        "The second operand must be greater than u128::MAX"
    );
    if *u.high() == 0 {
        unsafe { core::hint::unreachable_unchecked() }
    }

    #[inline]
    fn full_shl(a: &U256, shift: u32) -> [u128; 3] {
        debug_assert!(shift < N_UDWORD_BITS);
        let mut u = [0_u128; 3];
        let u_lo = a.low() << shift;
        let u_hi = a >> (N_UDWORD_BITS - shift);
        u[0] = u_lo;
        u[1] = *u_hi.low();
        u[2] = *u_hi.high();

        u
    }

    #[inline]
    fn full_shr(u: &[u128; 3], shift: u32) -> U256 {
        debug_assert!(shift < N_UDWORD_BITS);
        let mut res = U256::ZERO;
        *res.low_mut() = u[0] >> shift;
        *res.high_mut() = u[1] >> shift;
        // carry
        if shift > 0 {
            let sh = N_UDWORD_BITS - shift;
            *res.low_mut() |= u[1] << sh;
            *res.high_mut() |= u[2] << sh;
        }

        res
    }

    // returns (lo, hi)
    #[inline]
    const fn split_u128_to_u128(a: u128) -> (u128, u128) {
        (a & 0xFFFFFFFFFFFFFFFF, a >> (N_UDWORD_BITS / 2))
    }

    // returns (lo, hi)
    #[inline]
    const fn fullmul_u128(a: u128, b: u128) -> (u128, u128) {
        let (a0, a1) = split_u128_to_u128(a);
        let (b0, b1) = split_u128_to_u128(b);

        let mut t = a0 * b0;
        let mut k: u128;
        let w3: u128;
        (w3, k) = split_u128_to_u128(t);

        t = a1 * b0 + k;
        let (w1, w2) = split_u128_to_u128(t);
        t = a0 * b1 + w1;
        k = t >> 64;

        let w_hi = a1 * b1 + w2 + k;
        let w_lo = (t << 64) + w3;

        (w_lo, w_hi)
    }

    #[inline]
    fn fullmul_u256_u128(a: &U256, b: u128) -> [u128; 3] {
        let mut acc = [0_u128; 3];
        let mut lo: u128;
        let mut carry: u128;
        let c: bool;
        if b != 0 {
            (lo, carry) = fullmul_u128(*a.low(), b);
            acc[0] = lo;
            acc[1] = carry;
            (lo, carry) = fullmul_u128(*a.high(), b);
            (acc[1], c) = acc[1].overflowing_add(lo);
            acc[2] = carry + c as u128;
        }

        acc
    }

    #[inline]
    const fn add_carry(a: u128, b: u128, c: bool) -> (u128, bool) {
        let (res1, overflow1) = b.overflowing_add(c as u128);
        let (res2, overflow2) = u128::overflowing_add(a, res1);

        (res2, overflow1 || overflow2)
    }

    #[inline]
    const fn sub_carry(a: u128, b: u128, c: bool) -> (u128, bool) {
        let (res1, overflow1) = b.overflowing_add(c as u128);
        let (res2, overflow2) = u128::overflowing_sub(a, res1);

        (res2, overflow1 || overflow2)
    }

    // D1.
    // Make sure 128th bit in v's highest word is set.
    // If we shift both u and v, it won't affect the quotient
    // and the remainder will only need to be shifted back.
    let shift = v.high().leading_zeros();
    debug_assert!(shift < N_UDWORD_BITS);
    let v = v << shift;
    // u will store the remainder (shifted)
    let mut u = full_shl(u, shift);

    // quotient
    let mut q = U256::ZERO;
    let v_n_1 = *v.high();
    let v_n_2 = *v.low();

    if v_n_1 >> (N_UDWORD_BITS - 1) != 1 {
        debug_assert!(false);

        // SAFETY: `v_n_1` must be normalized because input `v` has
        // been checked to be non-zero.
        unsafe { core::hint::unreachable_unchecked() }
    }

    // D2. D7. - unrolled loop j == 0, n == 2, m == 0 (only one possible iteration)
    let mut r_hat: u128 = 0;
    let u_jn = u[2];

    // D3.
    // q_hat is our guess for the j-th quotient digit
    // q_hat = min(b - 1, (u_{j+n} * b + u_{j+n-1}) / v_{n-1})
    // b = 1 << WORD_BITS
    // Theorem B: q_hat >= q_j >= q_hat - 2
    let mut q_hat = if u_jn < v_n_1 {
        //let (mut q_hat, mut r_hat) = _div_mod_u128(u_jn, u[j + n - 1], v_n_1);
        let mut q_hat = udiv256_by_128_to_128(
            u_jn,
            u[1],
            unsafe { NonZeroU128::new_unchecked(v_n_1) },
            &mut r_hat,
        );
        let mut overflow: bool;
        // this loop takes at most 2 iterations
        loop {
            let another_iteration = {
                // check if q_hat * v_{n-2} > b * r_hat + u_{j+n-2}
                let (lo, hi) = fullmul_u128(q_hat, v_n_2);
                hi > r_hat || (hi == r_hat && lo > u[0])
            };
            if !another_iteration {
                break;
            }
            q_hat -= 1;
            (r_hat, overflow) = r_hat.overflowing_add(v_n_1);
            // if r_hat overflowed, we're done
            if overflow {
                break;
            }
        }
        q_hat
    } else {
        // here q_hat >= q_j >= q_hat - 1
        u128::MAX
    };

    // ex. 20:
    // since q_hat * v_{n-2} <= b * r_hat + u_{j+n-2},
    // either q_hat == q_j, or q_hat == q_j + 1

    // D4.
    // let's assume optimistically q_hat == q_j
    // subtract (q_hat * v) from u[j..]
    let q_hat_v = fullmul_u256_u128(&v, q_hat);
    // u[j..] -= q_hat_v;
    let mut c = false;
    (u[0], c) = sub_carry(u[0], q_hat_v[0], c);
    (u[1], c) = sub_carry(u[1], q_hat_v[1], c);
    (u[2], c) = sub_carry(u[2], q_hat_v[2], c);

    // D6.
    // actually, q_hat == q_j + 1 and u[j..] has overflowed
    // highly unlikely ~ (1 / 2^127)
    if c {
        q_hat -= 1;
        // add v to u[j..]
        c = false;
        (u[0], c) = add_carry(u[0], *v.low(), c);
        (u[1], c) = add_carry(u[1], *v.high(), c);
        u[2] = u[2].wrapping_add(c as u128);
    }

    // D5.
    *q.low_mut() = q_hat;

    // D8.
    let remainder = full_shr(&u, shift);

    (q, remainder)
}

#[inline]
pub fn udiv2(r: &mut U256, a: &U256) {
    let (a, b) = (*r, a);
    // SAFETY: `udivmod4` does not write `MaybeUninit::uninit()` to `res` and
    // `U256` does not implement `Drop`.
    let res = unsafe { &mut *(r as *mut U256).cast() };
    udivmod4(res, &a, b, None);
}

#[inline]
pub fn udiv3(r: &mut MaybeUninit<U256>, a: &U256, b: &U256) {
    udivmod4(r, a, b, None);
}

#[inline]
pub fn urem2(r: &mut U256, a: &U256) {
    let mut res = MaybeUninit::uninit();
    let (a, b) = (*r, a);
    // SAFETY: `udivmod4` does not write `MaybeUninit::uninit()` to `rem` and
    // `U256` does not implement `Drop`.
    let r = unsafe { &mut *(r as *mut U256).cast() };
    udivmod4(&mut res, &a, b, Some(r));
}

#[inline]
pub fn urem3(r: &mut MaybeUninit<U256>, a: &U256, b: &U256) {
    let mut res = MaybeUninit::uninit();
    udivmod4(&mut res, a, b, Some(r));
}

pub fn idivmod4(
    res: &mut MaybeUninit<I256>,
    a: &I256,
    b: &I256,
    mut rem: Option<&mut MaybeUninit<I256>>,
) {
    const BITS_IN_TWORD_M1: u32 = 255;
    let s_a = a >> BITS_IN_TWORD_M1; // s_a = a < 0 ? -1 : 0
    let mut s_b = b >> BITS_IN_TWORD_M1; // s_b = b < 0 ? -1 : 0
    let a = (a ^ s_a).wrapping_sub(s_a); // negate if s_a == -1
    let b = (b ^ s_b).wrapping_sub(s_b); // negate if s_b == -1
    s_b ^= s_a; // sign of quotient
    udivmod4(
        cast!(uninit: res),
        cast!(ref: &a),
        cast!(ref: &b),
        cast!(optuninit: rem),
    );
    let q = unsafe { res.assume_init_ref() };
    let q = (q ^ s_b).wrapping_sub(s_b); // negate if s_b == -1
    res.write(q);
    if let Some(rem) = rem {
        let r = unsafe { rem.assume_init_ref() };
        let r = (r ^ s_a).wrapping_sub(s_a);
        rem.write(r);
    }
}

#[inline]
pub fn idiv2(r: &mut I256, a: &I256) {
    let (a, b) = (*r, a);
    // SAFETY: `udivmod4` does not write `MaybeUninit::uninit()` to `res` and
    // `U256` does not implement `Drop`.
    let res = unsafe { &mut *(r as *mut I256).cast() };
    idivmod4(res, &a, b, None);
}

#[inline]
pub fn idiv3(r: &mut MaybeUninit<I256>, a: &I256, b: &I256) {
    idivmod4(r, a, b, None);
}

#[inline]
pub fn irem2(r: &mut I256, a: &I256) {
    let mut res = MaybeUninit::uninit();
    let (a, b) = (*r, a);
    // SAFETY: `udivmod4` does not write `MaybeUninit::uninit()` to `rem` and
    // `U256` does not implement `Drop`.
    let r = unsafe { &mut *(r as *mut I256).cast() };
    idivmod4(&mut res, &a, b, Some(r));
}

#[inline]
pub fn irem3(r: &mut MaybeUninit<I256>, a: &I256, b: &I256) {
    let mut res = MaybeUninit::uninit();
    idivmod4(&mut res, a, b, Some(r));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AsU256;

    fn udiv(a: impl AsU256, b: impl AsU256) -> U256 {
        let mut r = MaybeUninit::uninit();
        udiv3(&mut r, &a.as_u256(), &b.as_u256());
        unsafe { r.assume_init() }
    }

    fn urem(a: impl AsU256, b: impl AsU256) -> U256 {
        let mut r = MaybeUninit::uninit();
        urem3(&mut r, &a.as_u256(), &b.as_u256());
        unsafe { r.assume_init() }
    }

    #[test]
    fn division() {
        // 0 X
        // ---
        // 0 X
        assert_eq!(udiv(100, 9), 11);

        // 0 X
        // ---
        // K X
        assert_eq!(udiv(!0u128, U256::ONE << 128u32), 0);

        // K 0
        // ---
        // K 0
        assert_eq!(udiv(U256::from_words(100, 0), U256::from_words(10, 0)), 10);

        // K K
        // ---
        // K 0
        assert_eq!(udiv(U256::from_words(100, 1337), U256::ONE << 130u32), 25);
        assert_eq!(
            udiv(U256::from_words(1337, !0), U256::from_words(63, 0)),
            21
        );

        // K X
        // ---
        // 0 K
        assert_eq!(
            udiv(U256::from_words(42, 0), U256::ONE),
            U256::from_words(42, 0),
        );
        assert_eq!(
            udiv(U256::from_words(42, 42), U256::ONE << 42),
            42u128 << (128 - 42),
        );
        assert_eq!(
            udiv(U256::from_words(1337, !0), 0xc0ffee),
            35996389033280467545299711090127855,
        );
        assert_eq!(
            udiv(U256::from_words(42, 0), 99),
            144362216269489045105674075880144089708,
        );

        // K X
        // ---
        // K K
        assert_eq!(
            udiv(U256::from_words(100, 100), U256::from_words(1000, 1000)),
            0,
        );
        assert_eq!(
            udiv(U256::from_words(1337, !0), U256::from_words(43, !0)),
            30,
        );
    }

    #[test]
    #[should_panic]
    fn division_by_zero() {
        udiv(1, 0);
    }

    #[test]
    fn remainder() {
        // 0 X
        // ---
        // 0 X
        assert_eq!(urem(100, 9), 1);

        // 0 X
        // ---
        // K X
        assert_eq!(urem(!0u128, U256::ONE << 128u32), !0u128);

        // K 0
        // ---
        // K 0
        assert_eq!(urem(U256::from_words(100, 0), U256::from_words(10, 0)), 0);

        // K K
        // ---
        // K 0
        assert_eq!(urem(U256::from_words(100, 1337), U256::ONE << 130u32), 1337);
        assert_eq!(
            urem(U256::from_words(1337, !0), U256::from_words(63, 0)),
            U256::from_words(14, !0),
        );

        // K X
        // ---
        // 0 K
        assert_eq!(urem(U256::from_words(42, 0), U256::ONE), 0);
        assert_eq!(urem(U256::from_words(42, 42), U256::ONE << 42), 42);
        assert_eq!(urem(U256::from_words(1337, !0), 0xc0ffee), 1910477);
        assert_eq!(urem(U256::from_words(42, 0), 99), 60);

        // K X
        // ---
        // K K
        assert_eq!(
            urem(U256::from_words(100, 100), U256::from_words(1000, 1000)),
            U256::from_words(100, 100),
        );
        assert_eq!(
            urem(U256::from_words(1337, !0), U256::from_words(43, !0)),
            U256::from_words(18, 29),
        );
    }

    #[test]
    #[should_panic]
    fn remainder_by_zero() {
        urem(1, 0);
    }
}
