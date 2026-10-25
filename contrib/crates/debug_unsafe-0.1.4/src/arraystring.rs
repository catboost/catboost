use arraystring::{prelude::Capacity, ArrayString};

pub trait ArrayStringFrom: Sized {
    fn from_chars_safe_unchecked<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = char>;
}

impl<SIZE: Capacity> ArrayStringFrom for ArrayString<SIZE> {
    #[inline(always)]
    fn from_chars_safe_unchecked<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = char>,
    {
        if cfg!(debug_assertions) {
            Self::try_from_chars(iter).unwrap()
        } else {
            unsafe { Self::from_chars_unchecked(iter) }
        }
    }
}
