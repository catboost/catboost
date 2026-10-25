//! Module implementing formatting for `U256` type.

use crate::uint::U256;

impl_fmt! {
    impl Fmt for U256;
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn debug() {
        assert_eq!(
            format!("{:?}", U256::MAX),
            "115792089237316195423570985008687907853269984665640564039457584007913129639935",
        );
        assert_eq!(
            format!("{:x?}", U256::MAX),
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        );
        assert_eq!(
            format!("{:#X?}", U256::MAX),
            "0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        );
    }

    #[test]
    fn display() {
        assert_eq!(
            format!("{}", U256::MAX),
            "115792089237316195423570985008687907853269984665640564039457584007913129639935",
        );
    }

    #[test]
    fn radix() {
        assert_eq!(format!("{:b}", U256::new(42)), "101010");
        assert_eq!(format!("{:o}", U256::new(42)), "52");
        assert_eq!(format!("{:x}", U256::new(42)), "2a");
    }

    #[test]
    fn exp() {
        assert_eq!(format!("{:e}", U256::new(42)), "4.2e1");
        assert_eq!(format!("{:e}", U256::new(10).pow(77)), "1e77");
        assert_eq!(format!("{:E}", U256::new(10).pow(39) * 1337), "1.337E42");
    }
}
