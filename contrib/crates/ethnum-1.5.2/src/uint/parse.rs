//! Module implementing parsing for `U256` type.

use crate::uint::U256;

impl_from_str! {
    impl FromStr for U256;
}

pub const fn const_from_str_prefixed(src: &str) -> U256 {
    assert!(!src.is_empty(), "empty string");

    let bytes = src.as_bytes();
    let start = bytes[0] == b'+';
    crate::parse::const_from_str_prefixed(bytes, start as _)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::from_str_radix;
    use core::num::IntErrorKind;

    #[test]
    fn from_str() {
        assert_eq!("42".parse::<U256>().unwrap(), 42);
    }

    #[test]
    fn from_str_prefixed() {
        assert_eq!(from_str_radix::<U256>("0b101", 2, Some("0b")).unwrap(), 5);
        assert_eq!(from_str_radix::<U256>("0xf", 16, Some("0x")).unwrap(), 15);
    }

    #[test]
    fn from_str_errors() {
        assert_eq!(
            from_str_radix::<U256>("", 2, None).unwrap_err().kind(),
            &IntErrorKind::Empty,
        );
        assert_eq!(
            from_str_radix::<U256>("?", 2, None).unwrap_err().kind(),
            &IntErrorKind::InvalidDigit,
        );
        assert_eq!(
            from_str_radix::<U256>("1", 16, Some("0x"))
                .unwrap_err()
                .kind(),
            &IntErrorKind::InvalidDigit,
        );
        assert_eq!(
            from_str_radix::<U256>("-1", 10, None).unwrap_err().kind(),
            &IntErrorKind::InvalidDigit,
        );
        assert_eq!(
            from_str_radix::<U256>(
                "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
                36,
                None
            )
            .unwrap_err()
            .kind(),
            &IntErrorKind::PosOverflow,
        );
    }

    #[test]
    fn const_parse() {
        assert_eq!(const_from_str_prefixed("+0b1101"), 0b1101);
        assert_eq!(const_from_str_prefixed("0o777"), 0o777);
        assert_eq!(const_from_str_prefixed("+0x1f"), 0x1f);
        assert_eq!(const_from_str_prefixed("42"), 42);

        assert_eq!(
            const_from_str_prefixed(
                "0xffff_ffff_ffff_ffff_ffff_ffff_ffff_fffe\
                   baae_dce6_af48_a03b_bfd2_5e8c_d036_4141"
            ),
            U256::from_words(
                0xffff_ffff_ffff_ffff_ffff_ffff_ffff_fffe,
                0xbaae_dce6_af48_a03b_bfd2_5e8c_d036_4141,
            ),
        );

        assert_eq!(
            const_from_str_prefixed(
                "0x0000_0000_0000_0000_0000_0000_0000_0000\
                   0000_0000_0000_0000_0000_0000_0000_0000"
            ),
            U256::MIN,
        );
        assert_eq!(
            const_from_str_prefixed(
                "+0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff\
                    ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
            ),
            U256::MAX,
        );
    }

    #[test]
    #[should_panic]
    fn const_parse_overflow() {
        const_from_str_prefixed(
            "0x1\
               0000_0000_0000_0000_0000_0000_0000_0000\
               0000_0000_0000_0000_0000_0000_0000_0000",
        );
    }

    #[test]
    #[should_panic]
    fn const_parse_invalid() {
        const_from_str_prefixed("invalid");
    }
}
