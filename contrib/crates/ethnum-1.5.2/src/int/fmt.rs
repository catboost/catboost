//! Module implementing formatting for `I256` type.

use crate::int::I256;

impl_fmt! {
    impl Fmt for I256;
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn from_str() {
        assert_eq!("42".parse::<I256>().unwrap(), 42);
    }

    #[test]
    fn debug() {
        assert_eq!(
            format!("{:?}", I256::MAX),
            "57896044618658097711785492504343953926634992332820282019728792003956564819967",
        );
        assert_eq!(
            format!("{:x?}", I256::MIN),
            "8000000000000000000000000000000000000000000000000000000000000000",
        );
        assert_eq!(
            format!("{:#X?}", I256::MAX),
            "0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        );
    }

    #[test]
    fn display() {
        assert_eq!(
            format!("{}", I256::MIN),
            "-57896044618658097711785492504343953926634992332820282019728792003956564819968",
        );
        assert_eq!(
            format!(
                "{}",
                I256::from_words(0, -1329227995784915854529085396220968961)
            ),
            "338953138925153547608845522035547242495",
        );
    }

    #[test]
    fn radix() {
        assert_eq!(format!("{:b}", I256::new(42)), "101010");
        assert_eq!(format!("{:o}", I256::new(42)), "52");
        assert_eq!(format!("{:x}", I256::new(42)), "2a");

        // Note that there is no '-' sign for binary, octal or hex formatting!
        // This is the same behaviour for the standard iN types.
        assert_eq!(format!("{:b}", I256::MINUS_ONE), "1".repeat(256));
        assert_eq!(format!("{:o}", I256::MINUS_ONE), format!("{:7<86}", "1"));
        assert_eq!(format!("{:x}", I256::MINUS_ONE), "f".repeat(64));
    }

    #[test]
    fn exp() {
        assert_eq!(format!("{:e}", I256::new(42)), "4.2e1");
        assert_eq!(format!("{:e}", I256::new(10).pow(76)), "1e76");
        assert_eq!(format!("{:E}", -I256::new(10).pow(39) * 1337), "-1.337E42");
    }
}
