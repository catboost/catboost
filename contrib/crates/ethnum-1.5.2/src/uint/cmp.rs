//! Module with comparison implementations for `U256`.
//!
//! `PartialEq` is derived and not implemented, which is important for ensuring
//! that `match` can be used with `U256`.
//!
//! ```
//! # use ethnum::U256;
//! # let value = U256::new(42);
//!
//! match (value) {
//!     U256::ZERO => println!("I am zero"),
//!     U256::ONE => println!("I am one"),
//!     _ => println!("I am something else"),
//! }
//! ```
//!
//! `PartialEq` and `PartialOrd` implementations for `u128` are also provided
//! to allow notation such as:
//!
//! ```
//! # use ethnum::U256;
//!
//! assert_eq!(U256::new(42), 42);
//! assert!(U256::ONE > 0 && U256::ZERO == 0);
//! ```

use crate::uint::U256;
use core::cmp::Ordering;

impl Ord for U256 {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.into_words().cmp(&other.into_words())
    }
}

impl_cmp! {
    impl Cmp for U256 (u128);
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cmp::Ordering;

    #[test]
    fn cmp() {
        // 1e38
        let x = U256::from_words(0, 100000000000000000000000000000000000000);
        // 1e48
        let y = U256::from_words(2938735877, 18960114910927365649471927446130393088);
        assert!(x < y);
        assert_eq!(x.cmp(&y), Ordering::Less);
        assert!(y > x);
        assert_eq!(y.cmp(&x), Ordering::Greater);

        let x = U256::new(100);
        let y = U256::new(100);
        assert!(x <= y);
        assert_eq!(x.cmp(&y), Ordering::Equal);
    }
}
