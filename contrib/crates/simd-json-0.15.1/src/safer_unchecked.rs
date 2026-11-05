use core::slice::SliceIndex;

pub trait GetSaferUnchecked<T> {
    unsafe fn get_kinda_unchecked<I>(&self, index: I) -> &<I as SliceIndex<[T]>>::Output
    where
        I: SliceIndex<[T]>;

    unsafe fn get_kinda_unchecked_mut<I>(
        &mut self,
        index: I,
    ) -> &mut <I as SliceIndex<[T]>>::Output
    where
        I: SliceIndex<[T]>;
}

impl<T> GetSaferUnchecked<T> for [T] {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    unsafe fn get_kinda_unchecked<I>(&self, index: I) -> &<I as SliceIndex<[T]>>::Output
    where
        I: SliceIndex<[T]>,
    {
        #[cfg(debug_assertions)]
        let r = &self[index];
        #[cfg(not(debug_assertions))]
        let r = unsafe { self.get_unchecked(index) };
        r
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    unsafe fn get_kinda_unchecked_mut<I>(&mut self, index: I) -> &mut <I as SliceIndex<[T]>>::Output
    where
        I: SliceIndex<[T]>,
    {
        #[cfg(debug_assertions)]
        let r = &mut self[index];
        #[cfg(not(debug_assertions))]
        let r = unsafe { self.get_unchecked_mut(index) };
        r
    }
}
