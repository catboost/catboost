//! Implements the [`zeroize::Zeroize`] trait for [`CompactString`]

use crate::CompactString;
use zeroize::Zeroize;

#[cfg_attr(docsrs, doc(cfg(feature = "zeroize")))]
impl Zeroize for CompactString {
    fn zeroize(&mut self) {
        self.0.zeroize();
    }
}

#[cfg(test)]
mod tests {
    use std::string::ToString;

    use alloc::string::String;
    use test_strategy::proptest;

    use super::*;
    use crate::tests::rand_unicode;

    #[test]
    fn smoketest_zeroize() {
        let mut short = CompactString::from("hello");
        let mut control = short.as_str().to_string();
        short.zeroize();
        assert_eq!(short, "");
        control.zeroize();
        assert_eq!(control, short);

        let mut long = CompactString::from("I am a long string that will be on the heap");
        let mut control = long.as_str().to_string();
        long.zeroize();
        assert_eq!(long, "");
        assert!(long.is_heap_allocated());
        control.zeroize();
        assert_eq!(long, control);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn proptest_zeroize(#[strategy(rand_unicode())] s: String) {
        let mut compact = CompactString::new(s.clone());
        let mut control = s.clone();

        compact.zeroize();
        control.zeroize();

        assert_eq!(compact, control);
    }
}
