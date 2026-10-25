//! Module with casting helpers.

/// Cast references of `U256` to `I256` for intrinsic implementations.
macro_rules! cast {
    (mut: $x:expr) => {
        unsafe { &mut *($x as *mut $crate::int::I256).cast::<$crate::uint::U256>() }
    };
    (ref: $x:expr) => {
        unsafe { &*($x as *const $crate::int::I256).cast::<$crate::uint::U256>() }
    };
    (uninit: $x:expr) => {
        unsafe { &mut *($x).as_mut_ptr().cast::<::core::mem::MaybeUninit<$crate::uint::U256>>() }
    };
    (optuninit: $x:expr) => {
        ($x).as_mut().map(|x| cast!(uninit: *x))
    };
}
