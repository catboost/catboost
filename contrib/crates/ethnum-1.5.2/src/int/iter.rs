//! Module contains iterator specific trait implementations.
//!
//! ```
//! # use ethnum::I256;
//! assert_eq!((1..=3).map(I256::new).sum::<I256>(), 6);
//! assert_eq!([I256::new(6), I256::new(7)].iter().product::<I256>(), 42);
//! ```

use super::I256;

impl_iter! {
    impl Iter for I256;
}
