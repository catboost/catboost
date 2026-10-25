pub trait OptionUnwrapper<T>: Sized {
    fn unwrap_safe_unchecked(self) -> T;
}

impl<T> OptionUnwrapper<T> for Option<T> {
    #[inline(always)]
    fn unwrap_safe_unchecked(self) -> T {
        if cfg!(debug_assertions) {
            self.unwrap()
        } else {
            unsafe { self.unwrap_unchecked() }
        }
    }
}
