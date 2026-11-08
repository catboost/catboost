/// Derive Copy and Clone using the parameters (and bounds) as specified in []
macro_rules! copy_and_clone {
    ([$($parm:tt)*] $type_:ty) => {
        impl<$($parm)*> Copy for $type_ { }
        impl<$($parm)*> Clone for $type_ {
            #[inline(always)]
            fn clone(&self) -> Self { *self }
        }
    };
    ($type_:ty) => {
        copy_and_clone!{ [] $type_ }
    };
}

macro_rules! clone_bounds {
    ([$($parmbounds:tt)*] $typename:ident [$($parm:tt)*] {
        @copy {
            $($copyfield:ident,)*
        }
        $($field:ident,)*
    }) => {
        impl<$($parmbounds)*> Clone for $typename<$($parm)*> {
            fn clone(&self) -> Self {
                $typename {
                $(
                    $copyfield: self.$copyfield,
                )*
                $(
                    $field: self.$field.clone(),
                )*
                }
            }
        }
    };
}

/// This assertion is always enabled but only verbose (formatting when
/// debug assertions are enabled).
#[cfg(debug_assertions)]
macro_rules! ndassert {
    ($e:expr, $($t:tt)*) => { assert!($e, $($t)*) };
}

#[cfg(not(debug_assertions))]
macro_rules! ndassert {
    ($e:expr, $($_ignore:tt)*) => {
        assert!($e)
    };
}

macro_rules! expand_if {
    (@bool [true] $($body:tt)*) => { $($body)* };
    (@bool [false] $($body:tt)*) => { };
    (@nonempty [$($if_present:tt)+] $($body:tt)*) => {
        $($body)*
    };
    (@nonempty [] $($body:tt)*) => { };
}

// Macro to insert more informative out of bounds message in debug builds
#[cfg(debug_assertions)]
macro_rules! debug_bounds_check {
    ($self_:ident, $index:expr) => {
        if $index.index_checked(&$self_.dim, &$self_.strides).is_none() {
            panic!(
                "ndarray: index {:?} is out of bounds for array of shape {:?}",
                $index,
                $self_.shape()
            );
        }
    };
}

#[cfg(not(debug_assertions))]
macro_rules! debug_bounds_check {
    ($self_:ident, $index:expr) => {};
}
