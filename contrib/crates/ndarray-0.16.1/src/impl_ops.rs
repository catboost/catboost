// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::dimension::DimMax;
use crate::Zip;
use num_complex::Complex;

/// Elements that can be used as direct operands in arithmetic with arrays.
///
/// For example, `f64` is a `ScalarOperand` which means that for an array `a`,
/// arithmetic like `a + 1.0`, and, `a * 2.`, and `a += 3.` are allowed.
///
/// In the description below, let `A` be an array or array view,
/// let `B` be an array with owned data,
/// and let `C` be an array with mutable data.
///
/// `ScalarOperand` determines for which scalars `K` operations `&A @ K`, and `B @ K`,
/// and `C @= K` are defined, as ***right hand side operands***, for applicable
/// arithmetic operators (denoted `@`).
///
/// ***Left hand side*** scalar operands are not related to this trait
/// (they need one `impl` per concrete scalar type); but they are still
/// implemented for the same types, allowing operations
/// `K @ &A`, and `K @ B` for primitive numeric types `K`.
///
/// This trait ***does not*** limit which elements can be stored in an array in general.
/// Non-`ScalarOperand` types can still participate in arithmetic as array elements in
/// in array-array operations.
pub trait ScalarOperand: 'static + Clone {}
impl ScalarOperand for bool {}
impl ScalarOperand for i8 {}
impl ScalarOperand for u8 {}
impl ScalarOperand for i16 {}
impl ScalarOperand for u16 {}
impl ScalarOperand for i32 {}
impl ScalarOperand for u32 {}
impl ScalarOperand for i64 {}
impl ScalarOperand for u64 {}
impl ScalarOperand for i128 {}
impl ScalarOperand for u128 {}
impl ScalarOperand for isize {}
impl ScalarOperand for usize {}
impl ScalarOperand for f32 {}
impl ScalarOperand for f64 {}
impl ScalarOperand for Complex<f32> {}
impl ScalarOperand for Complex<f64> {}

macro_rules! impl_binary_op(
    ($trt:ident, $operator:tt, $mth:ident, $iop:tt, $doc:expr) => (
/// Perform elementwise
#[doc=$doc]
/// between `self` and `rhs`,
/// and return the result.
///
/// `self` must be an `Array` or `ArcArray`.
///
/// If their shapes disagree, `self` is broadcast to their broadcast shape.
///
/// **Panics** if broadcasting isn’t possible.
impl<A, B, S, S2, D, E> $trt<ArrayBase<S2, E>> for ArrayBase<S, D>
where
    A: Clone + $trt<B, Output=A>,
    B: Clone,
    S: DataOwned<Elem=A> + DataMut,
    S2: Data<Elem=B>,
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    type Output = ArrayBase<S, <D as DimMax<E>>::Output>;
    #[track_caller]
    fn $mth(self, rhs: ArrayBase<S2, E>) -> Self::Output
    {
        self.$mth(&rhs)
    }
}

/// Perform elementwise
#[doc=$doc]
/// between `self` and reference `rhs`,
/// and return the result.
///
/// `rhs` must be an `Array` or `ArcArray`.
///
/// If their shapes disagree, `self` is broadcast to their broadcast shape,
/// cloning the data if needed.
///
/// **Panics** if broadcasting isn’t possible.
impl<'a, A, B, S, S2, D, E> $trt<&'a ArrayBase<S2, E>> for ArrayBase<S, D>
where
    A: Clone + $trt<B, Output=A>,
    B: Clone,
    S: DataOwned<Elem=A> + DataMut,
    S2: Data<Elem=B>,
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    type Output = ArrayBase<S, <D as DimMax<E>>::Output>;
    #[track_caller]
    fn $mth(self, rhs: &ArrayBase<S2, E>) -> Self::Output
    {
        if self.ndim() == rhs.ndim() && self.shape() == rhs.shape() {
            let mut out = self.into_dimensionality::<<D as DimMax<E>>::Output>().unwrap();
            out.zip_mut_with_same_shape(rhs, clone_iopf(A::$mth));
            out
        } else {
            let (lhs_view, rhs_view) = self.broadcast_with(&rhs).unwrap();
            if lhs_view.shape() == self.shape() {
                let mut out = self.into_dimensionality::<<D as DimMax<E>>::Output>().unwrap();
                out.zip_mut_with_same_shape(&rhs_view, clone_iopf(A::$mth));
                out
            } else {
                Zip::from(&lhs_view).and(&rhs_view).map_collect_owned(clone_opf(A::$mth))
            }
        }
    }
}

/// Perform elementwise
#[doc=$doc]
/// between reference `self` and `rhs`,
/// and return the result.
///
/// `rhs` must be an `Array` or `ArcArray`.
///
/// If their shapes disagree, `self` is broadcast to their broadcast shape,
/// cloning the data if needed.
///
/// **Panics** if broadcasting isn’t possible.
impl<'a, A, B, S, S2, D, E> $trt<ArrayBase<S2, E>> for &'a ArrayBase<S, D>
where
    A: Clone + $trt<B, Output=B>,
    B: Clone,
    S: Data<Elem=A>,
    S2: DataOwned<Elem=B> + DataMut,
    D: Dimension,
    E: Dimension + DimMax<D>,
{
    type Output = ArrayBase<S2, <E as DimMax<D>>::Output>;
    #[track_caller]
    fn $mth(self, rhs: ArrayBase<S2, E>) -> Self::Output
    where
    {
        if self.ndim() == rhs.ndim() && self.shape() == rhs.shape() {
            let mut out = rhs.into_dimensionality::<<E as DimMax<D>>::Output>().unwrap();
            out.zip_mut_with_same_shape(self, clone_iopf_rev(A::$mth));
            out
        } else {
            let (rhs_view, lhs_view) = rhs.broadcast_with(self).unwrap();
            if rhs_view.shape() == rhs.shape() {
                let mut out = rhs.into_dimensionality::<<E as DimMax<D>>::Output>().unwrap();
                out.zip_mut_with_same_shape(&lhs_view, clone_iopf_rev(A::$mth));
                out
            } else {
                Zip::from(&lhs_view).and(&rhs_view).map_collect_owned(clone_opf(A::$mth))
            }
        }
    }
}

/// Perform elementwise
#[doc=$doc]
/// between references `self` and `rhs`,
/// and return the result as a new `Array`.
///
/// If their shapes disagree, `self` and `rhs` is broadcast to their broadcast shape,
/// cloning the data if needed.
///
/// **Panics** if broadcasting isn’t possible.
impl<'a, A, B, S, S2, D, E> $trt<&'a ArrayBase<S2, E>> for &'a ArrayBase<S, D>
where
    A: Clone + $trt<B, Output=A>,
    B: Clone,
    S: Data<Elem=A>,
    S2: Data<Elem=B>,
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    type Output = Array<A, <D as DimMax<E>>::Output>;
    #[track_caller]
    fn $mth(self, rhs: &'a ArrayBase<S2, E>) -> Self::Output {
        let (lhs, rhs) = if self.ndim() == rhs.ndim() && self.shape() == rhs.shape() {
            let lhs = self.view().into_dimensionality::<<D as DimMax<E>>::Output>().unwrap();
            let rhs = rhs.view().into_dimensionality::<<D as DimMax<E>>::Output>().unwrap();
            (lhs, rhs)
        } else {
            self.broadcast_with(rhs).unwrap()
        };
        Zip::from(lhs).and(rhs).map_collect(clone_opf(A::$mth))
    }
}

/// Perform elementwise
#[doc=$doc]
/// between `self` and the scalar `x`,
/// and return the result (based on `self`).
///
/// `self` must be an `Array` or `ArcArray`.
impl<A, S, D, B> $trt<B> for ArrayBase<S, D>
    where A: Clone + $trt<B, Output=A>,
          S: DataOwned<Elem=A> + DataMut,
          D: Dimension,
          B: ScalarOperand,
{
    type Output = ArrayBase<S, D>;
    fn $mth(mut self, x: B) -> ArrayBase<S, D> {
        self.map_inplace(move |elt| {
            *elt = elt.clone() $operator x.clone();
        });
        self
    }
}

/// Perform elementwise
#[doc=$doc]
/// between the reference `self` and the scalar `x`,
/// and return the result as a new `Array`.
impl<'a, A, S, D, B> $trt<B> for &'a ArrayBase<S, D>
    where A: Clone + $trt<B, Output=A>,
          S: Data<Elem=A>,
          D: Dimension,
          B: ScalarOperand,
{
    type Output = Array<A, D>;
    fn $mth(self, x: B) -> Self::Output {
        self.map(move |elt| elt.clone() $operator x.clone())
    }
}
    );
);

// Pick the expression $a for commutative and $b for ordered binop
macro_rules! if_commutative {
    (Commute { $a:expr } or { $b:expr }) => {
        $a
    };
    (Ordered { $a:expr } or { $b:expr }) => {
        $b
    };
}

macro_rules! impl_scalar_lhs_op {
    // $commutative flag. Reuse the self + scalar impl if we can.
    // We can do this safely since these are the primitive numeric types
    ($scalar:ty, $commutative:ident, $operator:tt, $trt:ident, $mth:ident, $doc:expr) => (
// these have no doc -- they are not visible in rustdoc
// Perform elementwise
// between the scalar `self` and array `rhs`,
// and return the result (based on `self`).
impl<S, D> $trt<ArrayBase<S, D>> for $scalar
    where S: DataOwned<Elem=$scalar> + DataMut,
          D: Dimension,
{
    type Output = ArrayBase<S, D>;
    fn $mth(self, rhs: ArrayBase<S, D>) -> ArrayBase<S, D> {
        if_commutative!($commutative {
            rhs.$mth(self)
        } or {{
            let mut rhs = rhs;
            rhs.map_inplace(move |elt| {
                *elt = self $operator *elt;
            });
            rhs
        }})
    }
}

// Perform elementwise
// between the scalar `self` and array `rhs`,
// and return the result as a new `Array`.
impl<'a, S, D> $trt<&'a ArrayBase<S, D>> for $scalar
    where S: Data<Elem=$scalar>,
          D: Dimension,
{
    type Output = Array<$scalar, D>;
    fn $mth(self, rhs: &ArrayBase<S, D>) -> Self::Output {
        if_commutative!($commutative {
            rhs.$mth(self)
        } or {
            rhs.map(move |elt| self.clone() $operator elt.clone())
        })
    }
}
    );
}

mod arithmetic_ops
{
    use super::*;
    use crate::imp_prelude::*;

    use std::ops::*;

    fn clone_opf<A: Clone, B: Clone, C>(f: impl Fn(A, B) -> C) -> impl FnMut(&A, &B) -> C
    {
        move |x, y| f(x.clone(), y.clone())
    }

    fn clone_iopf<A: Clone, B: Clone>(f: impl Fn(A, B) -> A) -> impl FnMut(&mut A, &B)
    {
        move |x, y| *x = f(x.clone(), y.clone())
    }

    fn clone_iopf_rev<A: Clone, B: Clone>(f: impl Fn(A, B) -> B) -> impl FnMut(&mut B, &A)
    {
        move |x, y| *x = f(y.clone(), x.clone())
    }

    impl_binary_op!(Add, +, add, +=, "addition");
    impl_binary_op!(Sub, -, sub, -=, "subtraction");
    impl_binary_op!(Mul, *, mul, *=, "multiplication");
    impl_binary_op!(Div, /, div, /=, "division");
    impl_binary_op!(Rem, %, rem, %=, "remainder");
    impl_binary_op!(BitAnd, &, bitand, &=, "bit and");
    impl_binary_op!(BitOr, |, bitor, |=, "bit or");
    impl_binary_op!(BitXor, ^, bitxor, ^=, "bit xor");
    impl_binary_op!(Shl, <<, shl, <<=, "left shift");
    impl_binary_op!(Shr, >>, shr, >>=, "right shift");

    macro_rules! all_scalar_ops {
        ($int_scalar:ty) => (
            impl_scalar_lhs_op!($int_scalar, Commute, +, Add, add, "addition");
            impl_scalar_lhs_op!($int_scalar, Ordered, -, Sub, sub, "subtraction");
            impl_scalar_lhs_op!($int_scalar, Commute, *, Mul, mul, "multiplication");
            impl_scalar_lhs_op!($int_scalar, Ordered, /, Div, div, "division");
            impl_scalar_lhs_op!($int_scalar, Ordered, %, Rem, rem, "remainder");
            impl_scalar_lhs_op!($int_scalar, Commute, &, BitAnd, bitand, "bit and");
            impl_scalar_lhs_op!($int_scalar, Commute, |, BitOr, bitor, "bit or");
            impl_scalar_lhs_op!($int_scalar, Commute, ^, BitXor, bitxor, "bit xor");
            impl_scalar_lhs_op!($int_scalar, Ordered, <<, Shl, shl, "left shift");
            impl_scalar_lhs_op!($int_scalar, Ordered, >>, Shr, shr, "right shift");
        );
    }
    all_scalar_ops!(i8);
    all_scalar_ops!(u8);
    all_scalar_ops!(i16);
    all_scalar_ops!(u16);
    all_scalar_ops!(i32);
    all_scalar_ops!(u32);
    all_scalar_ops!(i64);
    all_scalar_ops!(u64);
    all_scalar_ops!(isize);
    all_scalar_ops!(usize);
    all_scalar_ops!(i128);
    all_scalar_ops!(u128);

    impl_scalar_lhs_op!(bool, Commute, &, BitAnd, bitand, "bit and");
    impl_scalar_lhs_op!(bool, Commute, |, BitOr, bitor, "bit or");
    impl_scalar_lhs_op!(bool, Commute, ^, BitXor, bitxor, "bit xor");

    impl_scalar_lhs_op!(f32, Commute, +, Add, add, "addition");
    impl_scalar_lhs_op!(f32, Ordered, -, Sub, sub, "subtraction");
    impl_scalar_lhs_op!(f32, Commute, *, Mul, mul, "multiplication");
    impl_scalar_lhs_op!(f32, Ordered, /, Div, div, "division");
    impl_scalar_lhs_op!(f32, Ordered, %, Rem, rem, "remainder");

    impl_scalar_lhs_op!(f64, Commute, +, Add, add, "addition");
    impl_scalar_lhs_op!(f64, Ordered, -, Sub, sub, "subtraction");
    impl_scalar_lhs_op!(f64, Commute, *, Mul, mul, "multiplication");
    impl_scalar_lhs_op!(f64, Ordered, /, Div, div, "division");
    impl_scalar_lhs_op!(f64, Ordered, %, Rem, rem, "remainder");

    impl_scalar_lhs_op!(Complex<f32>, Commute, +, Add, add, "addition");
    impl_scalar_lhs_op!(Complex<f32>, Ordered, -, Sub, sub, "subtraction");
    impl_scalar_lhs_op!(Complex<f32>, Commute, *, Mul, mul, "multiplication");
    impl_scalar_lhs_op!(Complex<f32>, Ordered, /, Div, div, "division");

    impl_scalar_lhs_op!(Complex<f64>, Commute, +, Add, add, "addition");
    impl_scalar_lhs_op!(Complex<f64>, Ordered, -, Sub, sub, "subtraction");
    impl_scalar_lhs_op!(Complex<f64>, Commute, *, Mul, mul, "multiplication");
    impl_scalar_lhs_op!(Complex<f64>, Ordered, /, Div, div, "division");

    impl<A, S, D> Neg for ArrayBase<S, D>
    where
        A: Clone + Neg<Output = A>,
        S: DataOwned<Elem = A> + DataMut,
        D: Dimension,
    {
        type Output = Self;
        /// Perform an elementwise negation of `self` and return the result.
        fn neg(mut self) -> Self
        {
            self.map_inplace(|elt| {
                *elt = -elt.clone();
            });
            self
        }
    }

    impl<'a, A, S, D> Neg for &'a ArrayBase<S, D>
    where
        &'a A: 'a + Neg<Output = A>,
        S: Data<Elem = A>,
        D: Dimension,
    {
        type Output = Array<A, D>;
        /// Perform an elementwise negation of reference `self` and return the
        /// result as a new `Array`.
        fn neg(self) -> Array<A, D>
        {
            self.map(Neg::neg)
        }
    }

    impl<A, S, D> Not for ArrayBase<S, D>
    where
        A: Clone + Not<Output = A>,
        S: DataOwned<Elem = A> + DataMut,
        D: Dimension,
    {
        type Output = Self;
        /// Perform an elementwise unary not of `self` and return the result.
        fn not(mut self) -> Self
        {
            self.map_inplace(|elt| {
                *elt = !elt.clone();
            });
            self
        }
    }

    impl<'a, A, S, D> Not for &'a ArrayBase<S, D>
    where
        &'a A: 'a + Not<Output = A>,
        S: Data<Elem = A>,
        D: Dimension,
    {
        type Output = Array<A, D>;
        /// Perform an elementwise unary not of reference `self` and return the
        /// result as a new `Array`.
        fn not(self) -> Array<A, D>
        {
            self.map(Not::not)
        }
    }
}

mod assign_ops
{
    use super::*;
    use crate::imp_prelude::*;

    macro_rules! impl_assign_op {
        ($trt:ident, $method:ident, $doc:expr) => {
            use std::ops::$trt;

            #[doc=$doc]
            /// If their shapes disagree, `rhs` is broadcast to the shape of `self`.
            ///
            /// **Panics** if broadcasting isn’t possible.
            impl<'a, A, S, S2, D, E> $trt<&'a ArrayBase<S2, E>> for ArrayBase<S, D>
            where
                A: Clone + $trt<A>,
                S: DataMut<Elem = A>,
                S2: Data<Elem = A>,
                D: Dimension,
                E: Dimension,
            {
                #[track_caller]
                fn $method(&mut self, rhs: &ArrayBase<S2, E>) {
                    self.zip_mut_with(rhs, |x, y| {
                        x.$method(y.clone());
                    });
                }
            }

            #[doc=$doc]
            impl<A, S, D> $trt<A> for ArrayBase<S, D>
            where
                A: ScalarOperand + $trt<A>,
                S: DataMut<Elem = A>,
                D: Dimension,
            {
                fn $method(&mut self, rhs: A) {
                    self.map_inplace(move |elt| {
                        elt.$method(rhs.clone());
                    });
                }
            }
        };
    }

    impl_assign_op!(
        AddAssign,
        add_assign,
        "Perform `self += rhs` as elementwise addition (in place).\n"
    );
    impl_assign_op!(
        SubAssign,
        sub_assign,
        "Perform `self -= rhs` as elementwise subtraction (in place).\n"
    );
    impl_assign_op!(
        MulAssign,
        mul_assign,
        "Perform `self *= rhs` as elementwise multiplication (in place).\n"
    );
    impl_assign_op!(
        DivAssign,
        div_assign,
        "Perform `self /= rhs` as elementwise division (in place).\n"
    );
    impl_assign_op!(
        RemAssign,
        rem_assign,
        "Perform `self %= rhs` as elementwise remainder (in place).\n"
    );
    impl_assign_op!(
        BitAndAssign,
        bitand_assign,
        "Perform `self &= rhs` as elementwise bit and (in place).\n"
    );
    impl_assign_op!(
        BitOrAssign,
        bitor_assign,
        "Perform `self |= rhs` as elementwise bit or (in place).\n"
    );
    impl_assign_op!(
        BitXorAssign,
        bitxor_assign,
        "Perform `self ^= rhs` as elementwise bit xor (in place).\n"
    );
    impl_assign_op!(
        ShlAssign,
        shl_assign,
        "Perform `self <<= rhs` as elementwise left shift (in place).\n"
    );
    impl_assign_op!(
        ShrAssign,
        shr_assign,
        "Perform `self >>= rhs` as elementwise right shift (in place).\n"
    );
}
