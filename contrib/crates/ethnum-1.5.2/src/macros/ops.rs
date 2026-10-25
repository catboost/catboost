//! Module containing macros for implementing `core::ops` traits.

macro_rules! impl_ops {
    (
        for $int:ident | $prim:ident {
            add => $add2:ident, $add3:ident, $addc:ident;
            mul => $mul2:ident, $mul3:ident, $mulc:ident;
            sub => $sub2:ident, $sub3:ident, $subc:ident;
            div => $div2:ident, $div3:ident;
            rem => $rem2:ident, $rem3:ident;
            shl => $shl2:ident, $shl3:ident;
            shr => $shr2:ident, $shr3:ident;
        }
    ) => {
        __impl_ops_binop! {
            for $int | $prim

            impl Add {
                + add => $add3, $addc; "add with overflow"
            }
            impl Mul {
                * mul => $mul3, $mulc; "multiply with overflow"
            }
            impl Sub {
                - sub => $sub3, $subc; "subtract with overflow"
            }
        }

        __impl_ops_divmod! {
            for $int | $prim

            impl Div {
                / div => $div3; "divide by zero"
            }
            impl Rem {
                % rem => $rem3; "calculate the remainder with a divisor of zero"
            }
        }

        __impl_ops_shift! {
            for $int

            impl Shl {
                << shl => $shl3; "shift left with overflow"
            }
            impl Shr {
                >> shr => $shr3; "shift right with overflow"
            }
        }

        __impl_ops_unop! {
            impl Not for $int {
                not(x) {
                    let $int([a, b]) = x;
                    $int([!a, !b])
                }
            }
        }

        __impl_ops_bitwise! {
            for $int | $prim

            impl BitAnd {
                & bitand;
            }
            impl BitOr {
                | bitor;
            }
            impl BitXor {
                ^ bitxor;
            }
        }

        __impl_ops_binop_assign! {
            for $int | $prim

            impl AddAssign {
                += add_assign => $add2, +;
            }
            impl DivAssign {
                /= div_assign => $div2, /;
            }
            impl MulAssign {
                *= mul_assign => $mul2, *;
            }
            impl RemAssign {
                %= rem_assign => $rem2, %;
            }
            impl SubAssign {
                -= sub_assign => $sub2, -;
            }
        }

        __impl_ops_shift_assign! {
            for $int

            impl ShlAssign {
                <<= shl_assign => $shl2, <<;
            }
            impl ShrAssign {
                >>= shr_assign => $shr2, >>;
            }
        }

        __impl_ops_bitwise_assign! {
            for $int | $prim

            impl BitAndAssign {
                &= bitand_assign;
            }
            impl BitOrAssign {
                |= bitor_assign;
            }
            impl BitXorAssign {
                ^= bitxor_assign;
            }
        }
    };
}

macro_rules! impl_ops_neg {
    (
        for $int:ident {
            add => $add2:ident;
        }
    ) => {
        __impl_ops_unop! {
            impl Neg for $int {
                neg(x) {
                    #[cfg(debug_assertions)]
                    {
                        if x.eq(&I256::MIN) {
                            panic!("attempt to negate with overflow");
                        }
                    }
                    let mut result = !x;
                    $add2(&mut result, &$int::ONE);
                    result
                }
            }
        }
    };
}

macro_rules! __impl_ops_binop {
    (
        for $int:ident | $prim:ident
        $(
            impl $op:ident {
                $x:tt $method:ident => $op3:path, $opc:path; $msg:expr
            }
        )*
    ) => {$(
        impl ::core::ops::$op for &'_ $int {
            type Output = $int;

            #[inline]
            fn $method(self, rhs: Self) -> Self::Output {
                let mut result = ::core::mem::MaybeUninit::uninit();
                #[cfg(not(debug_assertions))]
                {
                    $op3(&mut result, self, rhs);
                }
                #[cfg(debug_assertions)]
                {
                    if $opc(&mut result, self, rhs) {
                        panic!(concat!("attempt to ", $msg));
                    }
                }
                unsafe { result.assume_init() }
            }
        }

        __impl_ops_binop_extra_variants! {
            impl $op for $int | $prim { $method = $x }
        }
    )*};
}

macro_rules! __impl_ops_binop_extra_variants {
    (
        impl $op:ident for $int:ident | $prim:ident { $method:ident = $x:tt }
    ) => {
        __impl_ops_binop_ref! {
            impl $op for $int {
                $method(a: &'_  $int, b:      $int) {  a $x &b };
                $method(a:      $int, b: &'_  $int) { &a $x  b };
                $method(a:      $int, b:      $int) { &a $x &b };

                $method(a: &'_  $int, b:     $prim) { a $x $int::new(b) };
                $method(a: &'_  $int, b: &'_ $prim) {  a $x *b };
                $method(a:      $int, b: &'_ $prim) { &a $x *b };
                $method(a:      $int, b:     $prim) { &a $x  b };

                $method(a:     $prim, b: &'_  $int) { $int::new(a) $x b };
                $method(a: &'_ $prim, b: &'_  $int) { *a $x  b };
                $method(a: &'_ $prim, b:      $int) { *a $x &b };
                $method(a:     $prim, b:      $int) {  a $x &b };
            }
        }
    };
}

macro_rules! __impl_ops_binop_ref {
    (
        impl $op:ident for $int:ident {$(
            $method:ident($lhs:ident: $lhst:ty, $rhs:ident: $rhst:ty) $impl:block;
        )*}
    ) => {$(
        impl ::core::ops::$op<$rhst> for $lhst {
            type Output = $int;

            #[inline]
            fn $method(self, rhs: $rhst) -> Self::Output {
                let ($lhs, $rhs) = (self, rhs);
                $impl
            }
        }
    )*}
}

macro_rules! __impl_ops_divmod {
    (
        for $int:ident | $prim:ident
        $(
            impl $op:ident {
                $x:tt $method:ident => $op3:path; $msg:expr
            }
        )*
    ) => {$(
        impl ::core::ops::$op for &'_ $int {
            type Output = $int;

            #[inline]
            fn $method(self, rhs: Self) -> Self::Output {
                if *rhs == 0 {
                    panic!(concat!("attempt to ", $msg));
                }

                let mut result = ::core::mem::MaybeUninit::uninit();
                $op3(&mut result, self, rhs);
                unsafe { result.assume_init() }
            }
        }

        __impl_ops_binop_extra_variants! {
            impl $op for $int | $prim { $method = $x }
        }
    )*};
}

macro_rules! __impl_ops_shift {
    (
        for $int:ident
        $(
            impl $op:ident {
                $x:tt $method:ident => $op3:path; $msg:expr
            }
        )*
    ) => {$(
        impl ::core::ops::$op<u32> for &'_ $int {
            type Output = $int;

            #[inline]
            fn $method(self, rhs: u32) -> Self::Output {
                #[cfg(debug_assertions)]
                if rhs > 0xff {
                    panic!(concat!("attempt to ", $msg));
                }

                let mut result = ::core::mem::MaybeUninit::uninit();
                $op3(&mut result, self, rhs);

                unsafe { result.assume_init() }
            }
        }

        __impl_ops_shift_extra_variants! {
            impl $op for $int { $method = $x }
        }
    )*};
}

macro_rules! __impl_ops_shift_extra_variants {
    (
        impl $op:ident for $int:ident { $method:ident = $x:tt }
    ) => {
        __impl_ops_binop_ref! {
            impl $op for $int {
                $method(a: &'_ $int, b: &'_ u32) {  a $x *b };
                $method(a:     $int, b: &'_ u32) { &a $x *b };
                $method(a:     $int, b:     u32) { &a $x  b };
            }
        }

        __impl_ops_shift_extra_variants! { __inner:
            impl $op<$crate::int::I256, $crate::uint::U256> for $int {
                $method = $x
                |b| { b.as_u32() }
            }
        }

        __impl_ops_shift_extra_variants! { __inner:
            impl $op<i8, i16, i32, i64, i128, isize, u8, u16, u64, u128, usize> for $int {
                $method = $x
                |b| { b as u32 }
            }
        }
    };

    (__inner:
        impl $op:ident <$($lhst:ty),*> for $int:ident
        { $method:ident = $x:tt |$lhs:ident| $conv:block }
    ) => {$(
        __impl_ops_binop_ref! {
            impl $op for $int {
                $method(a: &'_ $int, $lhs:     $lhst) {
                    #[cfg(not(debug_assertions))]
                    let b = $conv;
                    #[cfg(debug_assertions)]
                    let b = u32::try_from($lhs).unwrap_or(u32::MAX);
                    a $x b
                };
                $method(a: &'_ $int, b: &'_ $lhst) {  a $x *b };
                $method(a:     $int, b: &'_ $lhst) { &a $x *b };
                $method(a:     $int, b:     $lhst) { &a $x  b };
            }
        }
    )*};
}

macro_rules! __impl_ops_unop {
    (
        impl $op:ident for $int:ident {
            $method:ident($self:ident) $impl:block
        }
    ) => {
        impl ::core::ops::$op for $int {
            type Output = $int;

            #[inline]
            fn $method(self) -> Self::Output {
                let $self = self;
                $impl
            }
        }

        impl ::core::ops::$op for &'_ $int {
            type Output = $int;

            #[inline]
            fn $method(self) -> Self::Output {
                let $self = self;
                $impl
            }
        }
    };
}

macro_rules! __impl_ops_bitwise {
    (
        for $int:ident | $prim:ident
        $(
            impl $op:ident {
                $x:tt $method:ident;
            }
        )*
    ) => {$(
        impl ::core::ops::$op<&'_ $int> for &'_ $int {
            type Output = $int;

            #[inline]
            fn $method(self, rhs: &'_ $int) -> Self::Output {
                let $int([a0, a1]) = self;
                let $int([b0, b1]) = rhs;
                $int([a0 $x b0, a1 $x b1])
            }
        }

        __impl_ops_binop_extra_variants! {
            impl $op for $int | $prim { $method = $x }
        }
    )*};
}

macro_rules! __impl_ops_binop_assign {
    (
        for $int:ident | $prim:ident
        $(
            impl $op:ident {
                $x:tt $method:ident => $op2:path, $y:tt;
            }
        )*
    ) => {$(
        impl ::core::ops::$op<&'_ $int> for $int {
            #[inline]
            fn $method(&mut self, rhs: &'_ $int) {
                #[cfg(not(debug_assertions))]
                {
                    $op2(self, rhs);
                }
                #[cfg(debug_assertions)]
                {
                    *self = &*self $y rhs;
                }
            }
        }

        __impl_ops_binop_assign_extra_variants! {
            impl $op for $int | $prim { $method = $x }
        }
    )*};
}

macro_rules! __impl_ops_binop_assign_extra_variants {
    (
        impl $op:ident for $int:ident | $prim:ident { $method:ident = $x:tt }
    ) => {
        __impl_ops_binop_assign_ref! {
            impl $op for $int {
                $method(a, b:      $int) { *a $x &b };

                $method(a, b:     $prim) { *a $x $int::new(b) };
                $method(a, b: &'_ $prim) { *a $x *b };
            }
        }
    };
}

macro_rules! __impl_ops_binop_assign_ref {
    (
        impl $op:ident for $int:ident {$(
            $method:ident($self:ident, $rhs:ident: $rhst:ty) $impl:block;
        )*}
    ) => {$(
        impl ::core::ops::$op<$rhst> for $int {
            #[inline]
            fn $method(&mut self, rhs: $rhst) {
                let ($self, $rhs) = (self, rhs);
                $impl
            }
        }
    )*}
}

macro_rules! __impl_ops_shift_assign {
    (
        for $int:ident
        $(
            impl $op:ident {
                $x:tt $method:ident => $op2:path, $y:tt;
            }
        )*
    ) => {$(
        impl ::core::ops::$op<u32> for $int {
            #[inline]
            fn $method(&mut self, rhs: u32) {
                #[cfg(not(debug_assertions))]
                {
                    $op2(self, rhs);
                }
                #[cfg(debug_assertions)]
                {
                    *self = &*self $y rhs;
                }
            }
        }

        __impl_ops_shift_assign_extra_variants! {
            impl $op for $int { $method = $x }
        }
    )*};
}

macro_rules! __impl_ops_shift_assign_extra_variants {
    (
        impl $op:ident for $int:ident { $method:ident = $x:tt }
    ) => {
        __impl_ops_binop_assign_ref! {
            impl $op for $int {
                $method(a, b: &'_ u32) { *a $x *b };
            }
        }

        __impl_ops_shift_assign_extra_variants! { __inner:
            impl $op<$crate::int::I256, $crate::uint::U256> for $int {
                $method = $x
                |b| { b.as_u32() }
            }
        }

        __impl_ops_shift_assign_extra_variants! { __inner:
            impl $op<i8, i16, i32, i64, i128, isize, u8, u16, u64, u128, usize> for $int {
                $method = $x
                |b| { b as u32 }
            }
        }
    };

    (__inner:
        impl $op:ident <$($lhst:ty),*> for $int:ident
        { $method:ident = $x:tt |$lhs:ident| $conv:block }
    ) => {$(
        __impl_ops_binop_assign_ref! {
            impl $op for $int {
                $method(a, $lhs:     $lhst) {
                    #[cfg(not(debug_assertions))]
                    let b = $conv;
                    #[cfg(debug_assertions)]
                    let b = u32::try_from($lhs).unwrap_or(u32::MAX);
                    *a $x b
                };
                $method(a, b: &'_ $lhst) { *a $x *b };
            }
        }
    )*};
}

macro_rules! __impl_ops_bitwise_assign {
    (
        for $int:ident | $prim:ident
        $(
            impl $op:ident {
                $x:tt $method:ident;
            }
        )*
    ) => {$(
        impl ::core::ops::$op<&'_ $int> for $int {
            #[inline]
            fn $method(&mut self, rhs: &'_ $int) {
                let $int([a0, a1]) = self;
                let $int([b0, b1]) = rhs;
                *a0 $x b0;
                *a1 $x b1;
            }
        }

        __impl_ops_binop_assign_extra_variants! {
            impl $op for $int | $prim { $method = $x }
        }
    )*};
}
