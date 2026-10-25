//! Module implementing parsing for `I256` type.

use crate::int::I256;

impl_from_str! {
    impl FromStr for I256;
}

pub const fn const_from_str_prefixed(src: &str) -> I256 {
    assert!(!src.is_empty(), "empty string");

    let bytes = src.as_bytes();
    let (negate, start) = match bytes[0] {
        b'+' => (false, 1),
        b'-' => (true, 1),
        _ => (false, 0),
    };
    let uint = crate::parse::const_from_str_prefixed(bytes, start as _);

    let int = {
        let (hi, lo) = if negate {
            let (hi, lo) = uint.into_words();
            let (lo, carry) = (!lo).overflowing_add(1);
            let hi = (!hi).wrapping_add(carry as _);
            (hi, lo)
        } else {
            uint.into_words()
        };
        I256::from_words(hi as _, lo as _)
    };

    if matches!((negate, int.signum128()), (false, -1) | (true, 1)) {
        panic!("overflows integer type");
    }

    int
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::from_str_radix;
    use core::num::IntErrorKind;

    #[test]
    fn from_str() {
        assert_eq!("42".parse::<I256>().unwrap(), 42);
    }

    #[test]
    fn from_str_prefixed() {
        assert_eq!(from_str_radix::<I256>("0b101", 2, Some("0b")).unwrap(), 5);
        assert_eq!(from_str_radix::<I256>("-0xf", 16, Some("0x")).unwrap(), -15);
    }

    #[test]
    fn from_str_errors() {
        assert_eq!(
            from_str_radix::<I256>("", 2, None).unwrap_err().kind(),
            &IntErrorKind::Empty,
        );
        assert_eq!(
            from_str_radix::<I256>("?", 2, None).unwrap_err().kind(),
            &IntErrorKind::InvalidDigit,
        );
        assert_eq!(
            from_str_radix::<I256>("1", 16, Some("0x"))
                .unwrap_err()
                .kind(),
            &IntErrorKind::InvalidDigit,
        );
        assert_eq!(
            from_str_radix::<I256>(
                "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
                36,
                None
            )
            .unwrap_err()
            .kind(),
            &IntErrorKind::PosOverflow,
        );
        assert_eq!(
            from_str_radix::<I256>(
                "-zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
                36,
                None
            )
            .unwrap_err()
            .kind(),
            &IntErrorKind::NegOverflow,
        );
    }

    #[test]
    fn const_parse() {
        assert_eq!(const_from_str_prefixed("-0b1101"), -0b1101);
        assert_eq!(const_from_str_prefixed("0o777"), 0o777);
        assert_eq!(const_from_str_prefixed("-0x1f"), -0x1f);
        assert_eq!(const_from_str_prefixed("+42"), 42);

        assert_eq!(
            const_from_str_prefixed(
                "0x7fff_ffff_ffff_ffff_ffff_ffff_ffff_fffe\
                   baae_dce6_af48_a03b_bfd2_5e8c_d036_4141"
            ),
            I256::from_words(
                0x7fff_ffff_ffff_ffff_ffff_ffff_ffff_fffe,
                0xbaae_dce6_af48_a03b_bfd2_5e8c_d036_4141_u128 as _,
            ),
        );

        assert_eq!(
            const_from_str_prefixed(
                "-0x8000_0000_0000_0000_0000_0000_0000_0000\
                    0000_0000_0000_0000_0000_0000_0000_0000"
            ),
            I256::MIN,
        );
        assert_eq!(
            const_from_str_prefixed(
                "+0x7fff_ffff_ffff_ffff_ffff_ffff_ffff_ffff\
                    ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
            ),
            I256::MAX,
        );
    }

    #[test]
    #[should_panic]
    fn const_parse_overflow() {
        const_from_str_prefixed(
            "0x8000_0000_0000_0000_0000_0000_0000_0000\
               0000_0000_0000_0000_0000_0000_0000_0000",
        );
    }

    #[test]
    #[should_panic]
    fn const_parse_negative_overflow() {
        const_from_str_prefixed(
            "-0x8000_0000_0000_0000_0000_0000_0000_0000\
                0000_0000_0000_0000_0000_0000_0000_0001",
        );
    }

    #[test]
    #[should_panic]
    fn const_parse_invalid() {
        const_from_str_prefixed("invalid");
    }
}
