pub use self::inner::*;

pub trait AtomicMut<T> {
    fn read_mut(&mut self) -> T;
    fn write_mut(&mut self, value: T);
}

#[cfg(loom)]
mod inner {
    pub use loom::{cell, sync::atomic};

    impl super::AtomicMut<bool> for atomic::AtomicBool {
        fn read_mut(&mut self) -> bool {
            self.load(atomic::Ordering::Relaxed)
        }

        fn write_mut(&mut self, value: bool) {
            self.store(value, atomic::Ordering::Relaxed)
        }
    }

    impl<T> super::AtomicMut<*mut T> for atomic::AtomicPtr<T> {
        fn read_mut(&mut self) -> *mut T {
            self.load(atomic::Ordering::Relaxed)
        }

        fn write_mut(&mut self, value: *mut T) {
            self.store(value, atomic::Ordering::Relaxed)
        }
    }
}

#[cfg(not(loom))]
mod inner {
    pub use core::sync::atomic;

    impl super::AtomicMut<bool> for atomic::AtomicBool {
        #[inline(always)]
        fn read_mut(&mut self) -> bool {
            *self.get_mut()
        }

        #[inline(always)]
        fn write_mut(&mut self, value: bool) {
            *self.get_mut() = value;
        }
    }

    impl<T> super::AtomicMut<*mut T> for atomic::AtomicPtr<T> {
        #[inline(always)]
        fn read_mut(&mut self) -> *mut T {
            *self.get_mut()
        }

        #[inline(always)]
        fn write_mut(&mut self, value: *mut T) {
            *self.get_mut() = value;
        }
    }

    pub mod cell {
        pub struct UnsafeCell<T>(core::cell::UnsafeCell<T>);

        impl<T> UnsafeCell<T> {
            #[inline(always)]
            pub fn with<F, R>(&self, f: F) -> R
            where
                F: FnOnce(*const T) -> R,
            {
                f(self.0.get())
            }

            #[inline(always)]
            pub fn with_mut<F, R>(&self, f: F) -> R
            where
                F: FnOnce(*mut T) -> R,
            {
                f(self.0.get())
            }
        }
    }
}
