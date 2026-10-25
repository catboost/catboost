use ::core::slice::SliceIndex;

pub trait SliceGetter {
    fn get_safe_unchecked<I>(&self, index: I) -> &<I>::Output
    where
        I: SliceIndex<Self>;

    fn get_safe_unchecked_mut<I>(&mut self, index: I) -> &mut <I>::Output
    where
        I: SliceIndex<Self>;
}

impl<T> SliceGetter for [T] {
    #[inline(always)]
    fn get_safe_unchecked<I>(&self, index: I) -> &<I>::Output
    where
        I: SliceIndex<Self>,
    {
        if cfg!(debug_assertions) {
            &self[index]
        } else {
            unsafe { self.get_unchecked(index) }
        }
    }

    #[inline(always)]
    fn get_safe_unchecked_mut<I>(&mut self, index: I) -> &mut <I>::Output
    where
        I: SliceIndex<Self>,
    {
        if cfg!(debug_assertions) {
            &mut self[index]
        } else {
            unsafe { self.get_unchecked_mut(index) }
        }
    }
}
