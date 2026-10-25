//! Module with comparison implementations for `I256`.
//!
//! `PartialEq` is derived and not implemented, which is important for ensuring
//! that `match` can be used with `I256`.
//!
//! ```
//! # use ethnum::I256;
//! # let value = I256::new(42);
//!
//! match (value) {
//!     I256::ZERO => println!("I am zero"),
//!     I256::ONE => println!("I am one"),
//!     _ => println!("I am something else"),
//! }
//! ```
//!
//! `PartialEq` and `PartialOrd` implementations for `i128` are also provided
//! to allow notation such as:
//!
//! ```
//! # use ethnum::I256;
//! assert_eq!(I256::new(42), 42);
//! assert_eq!(42, I256::new(42));
//! assert!(I256::ONE > 0 && I256::ZERO == 0);
//! assert!(0 < I256::ONE && 0 == I256::ZERO);
//! ```

use super::I256;
use core::cmp::Ordering;

impl Ord for I256 {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        let (ahi, alo) = self.into_words();
        let (bhi, blo) = other.into_words();
        (ahi, alo as u128).cmp(&(bhi, blo as u128))
    }
}

impl_cmp! {
    impl Cmp for I256 (i128);
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cmp::Ordering;

    #[test]
    fn cmp() {
        // 1e38
        let x = I256::from_words(0, 100000000000000000000000000000000000000);
        // 1e48
        let y = I256::from_words(2938735877, 18960114910927365649471927446130393088);
        assert!(x < y);
        assert_eq!(x.cmp(&y), Ordering::Less);
        assert!(y > x);
        assert_eq!(y.cmp(&x), Ordering::Greater);

        let x = I256::new(100);
        let y = I256::new(100);
        assert!(x <= y);
        assert_eq!(x.cmp(&y), Ordering::Equal);

        assert!(I256::ZERO > I256::MIN);
        assert!(I256::ZERO < I256::MAX);

        assert!(I256::MAX > I256::MIN);
        assert!(I256::MIN < I256::MAX);
    }
}
