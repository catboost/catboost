// Copyright 2014-2020 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::imp_prelude::*;

#[cfg(feature = "blas")]
use crate::dimension::offset_from_low_addr_ptr_to_logical_ptr;
use crate::numeric_util;

use crate::{LinalgScalar, Zip};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use std::any::TypeId;
use std::mem::MaybeUninit;

use num_complex::Complex;
use num_complex::{Complex32 as c32, Complex64 as c64};

#[cfg(feature = "blas")]
use libc::c_int;

#[cfg(feature = "blas")]
use cblas_sys as blas_sys;
#[cfg(feature = "blas")]
use cblas_sys::{CblasNoTrans, CblasTrans, CBLAS_LAYOUT, CBLAS_TRANSPOSE};

/// len of vector before we use blas
#[cfg(feature = "blas")]
const DOT_BLAS_CUTOFF: usize = 32;
/// side of matrix before we use blas
#[cfg(feature = "blas")]
const GEMM_BLAS_CUTOFF: usize = 7;
#[cfg(feature = "blas")]
#[allow(non_camel_case_types)]
type blas_index = c_int; // blas index type

impl<A, S> ArrayBase<S, Ix1>
where S: Data<Elem = A>
{
    /// Perform dot product or matrix multiplication of arrays `self` and `rhs`.
    ///
    /// `Rhs` may be either a one-dimensional or a two-dimensional array.
    ///
    /// If `Rhs` is one-dimensional, then the operation is a vector dot
    /// product, which is the sum of the elementwise products (no conjugation
    /// of complex operands, and thus not their inner product). In this case,
    /// `self` and `rhs` must be the same length.
    ///
    /// If `Rhs` is two-dimensional, then the operation is matrix
    /// multiplication, where `self` is treated as a row vector. In this case,
    /// if `self` is shape *M*, then `rhs` is shape *M* × *N* and the result is
    /// shape *N*.
    ///
    /// **Panics** if the array shapes are incompatible.<br>
    /// *Note:* If enabled, uses blas `dot` for elements of `f32, f64` when memory
    /// layout allows.
    #[track_caller]
    pub fn dot<Rhs>(&self, rhs: &Rhs) -> <Self as Dot<Rhs>>::Output
    where Self: Dot<Rhs>
    {
        Dot::dot(self, rhs)
    }

    fn dot_generic<S2>(&self, rhs: &ArrayBase<S2, Ix1>) -> A
    where
        S2: Data<Elem = A>,
        A: LinalgScalar,
    {
        debug_assert_eq!(self.len(), rhs.len());
        assert!(self.len() == rhs.len());
        if let Some(self_s) = self.as_slice() {
            if let Some(rhs_s) = rhs.as_slice() {
                return numeric_util::unrolled_dot(self_s, rhs_s);
            }
        }
        let mut sum = A::zero();
        for i in 0..self.len() {
            unsafe {
                sum = sum + *self.uget(i) * *rhs.uget(i);
            }
        }
        sum
    }

    #[cfg(not(feature = "blas"))]
    fn dot_impl<S2>(&self, rhs: &ArrayBase<S2, Ix1>) -> A
    where
        S2: Data<Elem = A>,
        A: LinalgScalar,
    {
        self.dot_generic(rhs)
    }

    #[cfg(feature = "blas")]
    fn dot_impl<S2>(&self, rhs: &ArrayBase<S2, Ix1>) -> A
    where
        S2: Data<Elem = A>,
        A: LinalgScalar,
    {
        // Use only if the vector is large enough to be worth it
        if self.len() >= DOT_BLAS_CUTOFF {
            debug_assert_eq!(self.len(), rhs.len());
            assert!(self.len() == rhs.len());
            macro_rules! dot {
                ($ty:ty, $func:ident) => {{
                    if blas_compat_1d::<$ty, _>(self) && blas_compat_1d::<$ty, _>(rhs) {
                        unsafe {
                            let (lhs_ptr, n, incx) =
                                blas_1d_params(self.ptr.as_ptr(), self.len(), self.strides()[0]);
                            let (rhs_ptr, _, incy) =
                                blas_1d_params(rhs.ptr.as_ptr(), rhs.len(), rhs.strides()[0]);
                            let ret = blas_sys::$func(
                                n,
                                lhs_ptr as *const $ty,
                                incx,
                                rhs_ptr as *const $ty,
                                incy,
                            );
                            return cast_as::<$ty, A>(&ret);
                        }
                    }
                }};
            }

            dot! {f32, cblas_sdot};
            dot! {f64, cblas_ddot};
        }
        self.dot_generic(rhs)
    }
}

/// Return a pointer to the starting element in BLAS's view.
///
/// BLAS wants a pointer to the element with lowest address,
/// which agrees with our pointer for non-negative strides, but
/// is at the opposite end for negative strides.
#[cfg(feature = "blas")]
unsafe fn blas_1d_params<A>(ptr: *const A, len: usize, stride: isize) -> (*const A, blas_index, blas_index)
{
    // [x x x x]
    //        ^--ptr
    //        stride = -1
    //  ^--blas_ptr = ptr + (len - 1) * stride
    if stride >= 0 || len == 0 {
        (ptr, len as blas_index, stride as blas_index)
    } else {
        let ptr = ptr.offset((len - 1) as isize * stride);
        (ptr, len as blas_index, stride as blas_index)
    }
}

/// Matrix Multiplication
///
/// For two-dimensional arrays, the dot method computes the matrix
/// multiplication.
pub trait Dot<Rhs>
{
    /// The result of the operation.
    ///
    /// For two-dimensional arrays: a rectangular array.
    type Output;
    fn dot(&self, rhs: &Rhs) -> Self::Output;
}

impl<A, S, S2> Dot<ArrayBase<S2, Ix1>> for ArrayBase<S, Ix1>
where
    S: Data<Elem = A>,
    S2: Data<Elem = A>,
    A: LinalgScalar,
{
    type Output = A;

    /// Compute the dot product of one-dimensional arrays.
    ///
    /// The dot product is a sum of the elementwise products (no conjugation
    /// of complex operands, and thus not their inner product).
    ///
    /// **Panics** if the arrays are not of the same length.<br>
    /// *Note:* If enabled, uses blas `dot` for elements of `f32, f64` when memory
    /// layout allows.
    #[track_caller]
    fn dot(&self, rhs: &ArrayBase<S2, Ix1>) -> A
    {
        self.dot_impl(rhs)
    }
}

impl<A, S, S2> Dot<ArrayBase<S2, Ix2>> for ArrayBase<S, Ix1>
where
    S: Data<Elem = A>,
    S2: Data<Elem = A>,
    A: LinalgScalar,
{
    type Output = Array<A, Ix1>;

    /// Perform the matrix multiplication of the row vector `self` and
    /// rectangular matrix `rhs`.
    ///
    /// The array shapes must agree in the way that
    /// if `self` is *M*, then `rhs` is *M* × *N*.
    ///
    /// Return a result array with shape *N*.
    ///
    /// **Panics** if shapes are incompatible.
    #[track_caller]
    fn dot(&self, rhs: &ArrayBase<S2, Ix2>) -> Array<A, Ix1>
    {
        rhs.t().dot(self)
    }
}

impl<A, S> ArrayBase<S, Ix2>
where S: Data<Elem = A>
{
    /// Perform matrix multiplication of rectangular arrays `self` and `rhs`.
    ///
    /// `Rhs` may be either a one-dimensional or a two-dimensional array.
    ///
    /// If Rhs is two-dimensional, they array shapes must agree in the way that
    /// if `self` is *M* × *N*, then `rhs` is *N* × *K*.
    ///
    /// Return a result array with shape *M* × *K*.
    ///
    /// **Panics** if shapes are incompatible or the number of elements in the
    /// result would overflow `isize`.
    ///
    /// *Note:* If enabled, uses blas `gemv/gemm` for elements of `f32, f64`
    /// when memory layout allows. The default matrixmultiply backend
    /// is otherwise used for `f32, f64` for all memory layouts.
    ///
    /// ```
    /// use ndarray::arr2;
    ///
    /// let a = arr2(&[[1., 2.],
    ///                [0., 1.]]);
    /// let b = arr2(&[[1., 2.],
    ///                [2., 3.]]);
    ///
    /// assert!(
    ///     a.dot(&b) == arr2(&[[5., 8.],
    ///                         [2., 3.]])
    /// );
    /// ```
    #[track_caller]
    pub fn dot<Rhs>(&self, rhs: &Rhs) -> <Self as Dot<Rhs>>::Output
    where Self: Dot<Rhs>
    {
        Dot::dot(self, rhs)
    }
}

impl<A, S, S2> Dot<ArrayBase<S2, Ix2>> for ArrayBase<S, Ix2>
where
    S: Data<Elem = A>,
    S2: Data<Elem = A>,
    A: LinalgScalar,
{
    type Output = Array2<A>;
    fn dot(&self, b: &ArrayBase<S2, Ix2>) -> Array2<A>
    {
        let a = self.view();
        let b = b.view();
        let ((m, k), (k2, n)) = (a.dim(), b.dim());
        if k != k2 || m.checked_mul(n).is_none() {
            dot_shape_error(m, k, k2, n);
        }

        let lhs_s0 = a.strides()[0];
        let rhs_s0 = b.strides()[0];
        let column_major = lhs_s0 == 1 && rhs_s0 == 1;
        // A is Copy so this is safe
        let mut v = Vec::with_capacity(m * n);
        let mut c;
        unsafe {
            v.set_len(m * n);
            c = Array::from_shape_vec_unchecked((m, n).set_f(column_major), v);
        }
        mat_mul_impl(A::one(), &a, &b, A::zero(), &mut c.view_mut());
        c
    }
}

/// Assumes that `m` and `n` are ≤ `isize::MAX`.
#[cold]
#[inline(never)]
fn dot_shape_error(m: usize, k: usize, k2: usize, n: usize) -> !
{
    match m.checked_mul(n) {
        Some(len) if len <= isize::MAX as usize => {}
        _ => panic!("ndarray: shape {} × {} overflows isize", m, n),
    }
    panic!(
        "ndarray: inputs {} × {} and {} × {} are not compatible for matrix multiplication",
        m, k, k2, n
    );
}

#[cold]
#[inline(never)]
fn general_dot_shape_error(m: usize, k: usize, k2: usize, n: usize, c1: usize, c2: usize) -> !
{
    panic!("ndarray: inputs {} × {}, {} × {}, and output {} × {} are not compatible for matrix multiplication",
           m, k, k2, n, c1, c2);
}

/// Perform the matrix multiplication of the rectangular array `self` and
/// column vector `rhs`.
///
/// The array shapes must agree in the way that
/// if `self` is *M* × *N*, then `rhs` is *N*.
///
/// Return a result array with shape *M*.
///
/// **Panics** if shapes are incompatible.
impl<A, S, S2> Dot<ArrayBase<S2, Ix1>> for ArrayBase<S, Ix2>
where
    S: Data<Elem = A>,
    S2: Data<Elem = A>,
    A: LinalgScalar,
{
    type Output = Array<A, Ix1>;
    #[track_caller]
    fn dot(&self, rhs: &ArrayBase<S2, Ix1>) -> Array<A, Ix1>
    {
        let ((m, a), n) = (self.dim(), rhs.dim());
        if a != n {
            dot_shape_error(m, a, n, 1);
        }

        // Avoid initializing the memory in vec -- set it during iteration
        unsafe {
            let mut c = Array1::uninit(m);
            general_mat_vec_mul_impl(A::one(), self, rhs, A::zero(), c.raw_view_mut().cast::<A>());
            c.assume_init()
        }
    }
}

impl<A, S, D> ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    /// Perform the operation `self += alpha * rhs` efficiently, where
    /// `alpha` is a scalar and `rhs` is another array. This operation is
    /// also known as `axpy` in BLAS.
    ///
    /// If their shapes disagree, `rhs` is broadcast to the shape of `self`.
    ///
    /// **Panics** if broadcasting isn’t possible.
    #[track_caller]
    pub fn scaled_add<S2, E>(&mut self, alpha: A, rhs: &ArrayBase<S2, E>)
    where
        S: DataMut,
        S2: Data<Elem = A>,
        A: LinalgScalar,
        E: Dimension,
    {
        self.zip_mut_with(rhs, move |y, &x| *y = *y + (alpha * x));
    }
}

// mat_mul_impl uses ArrayView arguments to send all array kinds into
// the same instantiated implementation.
#[cfg(not(feature = "blas"))]
use self::mat_mul_general as mat_mul_impl;

#[cfg(feature = "blas")]
fn mat_mul_impl<A>(alpha: A, a: &ArrayView2<'_, A>, b: &ArrayView2<'_, A>, beta: A, c: &mut ArrayViewMut2<'_, A>)
where A: LinalgScalar
{
    let ((m, k), (k2, n)) = (a.dim(), b.dim());
    debug_assert_eq!(k, k2);
    if (m > GEMM_BLAS_CUTOFF || n > GEMM_BLAS_CUTOFF || k > GEMM_BLAS_CUTOFF)
        && (same_type::<A, f32>() || same_type::<A, f64>() || same_type::<A, c32>() || same_type::<A, c64>())
    {
        // Compute A B -> C
        // We require for BLAS compatibility that:
        // A, B, C are contiguous (stride=1) in their fastest dimension,
        // but they can be either row major/"c" or col major/"f".
        //
        // The "normal case" is CblasRowMajor for cblas.
        // Select CblasRowMajor / CblasColMajor to fit C's memory order.
        //
        // Apply transpose to A, B as needed if they differ from the row major case.
        // If C is CblasColMajor then transpose both A, B (again!)

        if let (Some(a_layout), Some(b_layout), Some(c_layout)) =
            (get_blas_compatible_layout(a), get_blas_compatible_layout(b), get_blas_compatible_layout(c))
        {
            let cblas_layout = c_layout.to_cblas_layout();
            let a_trans = a_layout.to_cblas_transpose_for(cblas_layout);
            let lda = blas_stride(&a, a_layout);

            let b_trans = b_layout.to_cblas_transpose_for(cblas_layout);
            let ldb = blas_stride(&b, b_layout);

            let ldc = blas_stride(&c, c_layout);

            macro_rules! gemm_scalar_cast {
                (f32, $var:ident) => {
                    cast_as(&$var)
                };
                (f64, $var:ident) => {
                    cast_as(&$var)
                };
                (c32, $var:ident) => {
                    &$var as *const A as *const _
                };
                (c64, $var:ident) => {
                    &$var as *const A as *const _
                };
            }

            macro_rules! gemm {
                ($ty:tt, $gemm:ident) => {
                    if same_type::<A, $ty>() {
                        // gemm is C ← αA^Op B^Op + βC
                        // Where Op is notrans/trans/conjtrans
                        unsafe {
                            blas_sys::$gemm(
                                cblas_layout,
                                a_trans,
                                b_trans,
                                m as blas_index,                 // m, rows of Op(a)
                                n as blas_index,                 // n, cols of Op(b)
                                k as blas_index,                 // k, cols of Op(a)
                                gemm_scalar_cast!($ty, alpha),   // alpha
                                a.ptr.as_ptr() as *const _,      // a
                                lda,                             // lda
                                b.ptr.as_ptr() as *const _,      // b
                                ldb,                             // ldb
                                gemm_scalar_cast!($ty, beta),    // beta
                                c.ptr.as_ptr() as *mut _,        // c
                                ldc,                             // ldc
                            );
                        }
                        return;
                    }
                };
            }

            gemm!(f32, cblas_sgemm);
            gemm!(f64, cblas_dgemm);
            gemm!(c32, cblas_cgemm);
            gemm!(c64, cblas_zgemm);

            unreachable!() // we checked above that A is one of f32, f64, c32, c64
        }
    }
    mat_mul_general(alpha, a, b, beta, c)
}

/// C ← α A B + β C
fn mat_mul_general<A>(
    alpha: A, lhs: &ArrayView2<'_, A>, rhs: &ArrayView2<'_, A>, beta: A, c: &mut ArrayViewMut2<'_, A>,
) where A: LinalgScalar
{
    let ((m, k), (_, n)) = (lhs.dim(), rhs.dim());

    // common parameters for gemm
    let ap = lhs.as_ptr();
    let bp = rhs.as_ptr();
    let cp = c.as_mut_ptr();
    let (rsc, csc) = (c.strides()[0], c.strides()[1]);
    if same_type::<A, f32>() {
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                cast_as(&alpha),
                ap as *const _,
                lhs.strides()[0],
                lhs.strides()[1],
                bp as *const _,
                rhs.strides()[0],
                rhs.strides()[1],
                cast_as(&beta),
                cp as *mut _,
                rsc,
                csc,
            );
        }
    } else if same_type::<A, f64>() {
        unsafe {
            matrixmultiply::dgemm(
                m,
                k,
                n,
                cast_as(&alpha),
                ap as *const _,
                lhs.strides()[0],
                lhs.strides()[1],
                bp as *const _,
                rhs.strides()[0],
                rhs.strides()[1],
                cast_as(&beta),
                cp as *mut _,
                rsc,
                csc,
            );
        }
    } else if same_type::<A, c32>() {
        unsafe {
            matrixmultiply::cgemm(
                matrixmultiply::CGemmOption::Standard,
                matrixmultiply::CGemmOption::Standard,
                m,
                k,
                n,
                complex_array(cast_as(&alpha)),
                ap as *const _,
                lhs.strides()[0],
                lhs.strides()[1],
                bp as *const _,
                rhs.strides()[0],
                rhs.strides()[1],
                complex_array(cast_as(&beta)),
                cp as *mut _,
                rsc,
                csc,
            );
        }
    } else if same_type::<A, c64>() {
        unsafe {
            matrixmultiply::zgemm(
                matrixmultiply::CGemmOption::Standard,
                matrixmultiply::CGemmOption::Standard,
                m,
                k,
                n,
                complex_array(cast_as(&alpha)),
                ap as *const _,
                lhs.strides()[0],
                lhs.strides()[1],
                bp as *const _,
                rhs.strides()[0],
                rhs.strides()[1],
                complex_array(cast_as(&beta)),
                cp as *mut _,
                rsc,
                csc,
            );
        }
    } else {
        // It's a no-op if `c` has zero length.
        if c.is_empty() {
            return;
        }

        // initialize memory if beta is zero
        if beta.is_zero() {
            c.fill(beta);
        }

        let mut i = 0;
        let mut j = 0;
        loop {
            unsafe {
                let elt = c.uget_mut((i, j));
                *elt =
                    *elt * beta + alpha * (0..k).fold(A::zero(), move |s, x| s + *lhs.uget((i, x)) * *rhs.uget((x, j)));
            }
            j += 1;
            if j == n {
                j = 0;
                i += 1;
                if i == m {
                    break;
                }
            }
        }
    }
}

/// General matrix-matrix multiplication.
///
/// Compute C ← α A B + β C
///
/// The array shapes must agree in the way that
/// if `a` is *M* × *N*, then `b` is *N* × *K* and `c` is *M* × *K*.
///
/// ***Panics*** if array shapes are not compatible<br>
/// *Note:* If enabled, uses blas `gemm` for elements of `f32, f64` when memory
/// layout allows.  The default matrixmultiply backend is otherwise used for
/// `f32, f64` for all memory layouts.
#[track_caller]
pub fn general_mat_mul<A, S1, S2, S3>(
    alpha: A, a: &ArrayBase<S1, Ix2>, b: &ArrayBase<S2, Ix2>, beta: A, c: &mut ArrayBase<S3, Ix2>,
) where
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
    S3: DataMut<Elem = A>,
    A: LinalgScalar,
{
    let ((m, k), (k2, n)) = (a.dim(), b.dim());
    let (m2, n2) = c.dim();
    if k != k2 || m != m2 || n != n2 {
        general_dot_shape_error(m, k, k2, n, m2, n2);
    } else {
        mat_mul_impl(alpha, &a.view(), &b.view(), beta, &mut c.view_mut());
    }
}

/// General matrix-vector multiplication.
///
/// Compute y ← α A x + β y
///
/// where A is a *M* × *N* matrix and x is an *N*-element column vector and
/// y an *M*-element column vector (one dimensional arrays).
///
/// ***Panics*** if array shapes are not compatible<br>
/// *Note:* If enabled, uses blas `gemv` for elements of `f32, f64` when memory
/// layout allows.
#[track_caller]
#[allow(clippy::collapsible_if)]
pub fn general_mat_vec_mul<A, S1, S2, S3>(
    alpha: A, a: &ArrayBase<S1, Ix2>, x: &ArrayBase<S2, Ix1>, beta: A, y: &mut ArrayBase<S3, Ix1>,
) where
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
    S3: DataMut<Elem = A>,
    A: LinalgScalar,
{
    unsafe { general_mat_vec_mul_impl(alpha, a, x, beta, y.raw_view_mut()) }
}

/// General matrix-vector multiplication
///
/// Use a raw view for the destination vector, so that it can be uninitialized.
///
/// ## Safety
///
/// The caller must ensure that the raw view is valid for writing.
/// the destination may be uninitialized iff beta is zero.
#[allow(clippy::collapsible_else_if)]
unsafe fn general_mat_vec_mul_impl<A, S1, S2>(
    alpha: A, a: &ArrayBase<S1, Ix2>, x: &ArrayBase<S2, Ix1>, beta: A, y: RawArrayViewMut<A, Ix1>,
) where
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
    A: LinalgScalar,
{
    let ((m, k), k2) = (a.dim(), x.dim());
    let m2 = y.dim();
    if k != k2 || m != m2 {
        general_dot_shape_error(m, k, k2, 1, m2, 1);
    } else {
        #[cfg(feature = "blas")]
        macro_rules! gemv {
            ($ty:ty, $gemv:ident) => {
                if same_type::<A, $ty>() {
                    if let Some(layout) = get_blas_compatible_layout(&a) {
                        if blas_compat_1d::<$ty, _>(&x) && blas_compat_1d::<$ty, _>(&y) {
                            // Determine stride between rows or columns. Note that the stride is
                            // adjusted to at least `k` or `m` to handle the case of a matrix with a
                            // trivial (length 1) dimension, since the stride for the trivial dimension
                            // may be arbitrary.
                            let a_trans = CblasNoTrans;

                            let a_stride = blas_stride(&a, layout);
                            let cblas_layout = layout.to_cblas_layout();

                            // Low addr in memory pointers required for x, y
                            let x_offset = offset_from_low_addr_ptr_to_logical_ptr(&x.dim, &x.strides);
                            let x_ptr = x.ptr.as_ptr().sub(x_offset);
                            let y_offset = offset_from_low_addr_ptr_to_logical_ptr(&y.dim, &y.strides);
                            let y_ptr = y.ptr.as_ptr().sub(y_offset);

                            let x_stride = x.strides()[0] as blas_index;
                            let y_stride = y.strides()[0] as blas_index;

                            blas_sys::$gemv(
                                cblas_layout,
                                a_trans,
                                m as blas_index,            // m, rows of Op(a)
                                k as blas_index,            // n, cols of Op(a)
                                cast_as(&alpha),            // alpha
                                a.ptr.as_ptr() as *const _, // a
                                a_stride,                   // lda
                                x_ptr as *const _,          // x
                                x_stride,
                                cast_as(&beta),             // beta
                                y_ptr as *mut _,            // y
                                y_stride,
                            );
                            return;
                        }
                    }
                }
            };
        }
        #[cfg(feature = "blas")]
        gemv!(f32, cblas_sgemv);
        #[cfg(feature = "blas")]
        gemv!(f64, cblas_dgemv);

        /* general */

        if beta.is_zero() {
            // when beta is zero, c may be uninitialized
            Zip::from(a.outer_iter()).and(y).for_each(|row, elt| {
                elt.write(row.dot(x) * alpha);
            });
        } else {
            Zip::from(a.outer_iter()).and(y).for_each(|row, elt| {
                *elt = *elt * beta + row.dot(x) * alpha;
            });
        }
    }
}

/// Kronecker product of 2D matrices.
///
/// The kronecker product of a LxN matrix A and a MxR matrix B is a (L*M)x(N*R)
/// matrix K formed by the block multiplication A_ij * B.
pub fn kron<A, S1, S2>(a: &ArrayBase<S1, Ix2>, b: &ArrayBase<S2, Ix2>) -> Array<A, Ix2>
where
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
    A: LinalgScalar,
{
    let dimar = a.shape()[0];
    let dimac = a.shape()[1];
    let dimbr = b.shape()[0];
    let dimbc = b.shape()[1];
    let mut out: Array2<MaybeUninit<A>> = Array2::uninit((
        dimar
            .checked_mul(dimbr)
            .expect("Dimensions of kronecker product output array overflows usize."),
        dimac
            .checked_mul(dimbc)
            .expect("Dimensions of kronecker product output array overflows usize."),
    ));
    Zip::from(out.exact_chunks_mut((dimbr, dimbc)))
        .and(a)
        .for_each(|out, &a| {
            Zip::from(out).and(b).for_each(|out, &b| {
                *out = MaybeUninit::new(a * b);
            })
        });
    unsafe { out.assume_init() }
}

#[inline(always)]
/// Return `true` if `A` and `B` are the same type
fn same_type<A: 'static, B: 'static>() -> bool
{
    TypeId::of::<A>() == TypeId::of::<B>()
}

// Read pointer to type `A` as type `B`.
//
// **Panics** if `A` and `B` are not the same type
fn cast_as<A: 'static + Copy, B: 'static + Copy>(a: &A) -> B
{
    assert!(same_type::<A, B>(), "expect type {} and {} to match",
            std::any::type_name::<A>(), std::any::type_name::<B>());
    unsafe { ::std::ptr::read(a as *const _ as *const B) }
}

/// Return the complex in the form of an array [re, im]
#[inline]
fn complex_array<A: 'static + Copy>(z: Complex<A>) -> [A; 2]
{
    [z.re, z.im]
}

#[cfg(feature = "blas")]
fn blas_compat_1d<A, S>(a: &ArrayBase<S, Ix1>) -> bool
where
    S: RawData,
    A: 'static,
    S::Elem: 'static,
{
    if !same_type::<A, S::Elem>() {
        return false;
    }
    if a.len() > blas_index::MAX as usize {
        return false;
    }
    let stride = a.strides()[0];
    if stride == 0 || stride > blas_index::MAX as isize || stride < blas_index::MIN as isize {
        return false;
    }
    true
}

#[cfg(feature = "blas")]
#[derive(Copy, Clone)]
#[cfg_attr(test, derive(PartialEq, Eq, Debug))]
enum BlasOrder
{
    C,
    F,
}

#[cfg(feature = "blas")]
impl BlasOrder
{
    fn transpose(self) -> Self
    {
        match self {
            Self::C => Self::F,
            Self::F => Self::C,
        }
    }

    #[inline]
    /// Axis of leading stride (opposite of contiguous axis)
    fn get_blas_lead_axis(self) -> usize
    {
        match self {
            Self::C => 0,
            Self::F => 1,
        }
    }

    fn to_cblas_layout(self) -> CBLAS_LAYOUT
    {
        match self {
            Self::C => CBLAS_LAYOUT::CblasRowMajor,
            Self::F => CBLAS_LAYOUT::CblasColMajor,
        }
    }

    /// When using cblas_sgemm (etc) with C matrix using `for_layout`,
    /// how should this `self` matrix be transposed
    fn to_cblas_transpose_for(self, for_layout: CBLAS_LAYOUT) -> CBLAS_TRANSPOSE
    {
        let effective_order = match for_layout {
            CBLAS_LAYOUT::CblasRowMajor => self,
            CBLAS_LAYOUT::CblasColMajor => self.transpose(),
        };

        match effective_order {
            Self::C => CblasNoTrans,
            Self::F => CblasTrans,
        }
    }
}

#[cfg(feature = "blas")]
fn is_blas_2d(dim: &Ix2, stride: &Ix2, order: BlasOrder) -> bool
{
    let (m, n) = dim.into_pattern();
    let s0 = stride[0] as isize;
    let s1 = stride[1] as isize;
    let (inner_stride, outer_stride, inner_dim, outer_dim) = match order {
        BlasOrder::C => (s1, s0, m, n),
        BlasOrder::F => (s0, s1, n, m),
    };

    if !(inner_stride == 1 || outer_dim == 1) {
        return false;
    }

    if s0 < 1 || s1 < 1 {
        return false;
    }

    if (s0 > blas_index::MAX as isize || s0 < blas_index::MIN as isize)
        || (s1 > blas_index::MAX as isize || s1 < blas_index::MIN as isize)
    {
        return false;
    }

    // leading stride must >= the dimension (no broadcasting/aliasing)
    if inner_dim > 1 && (outer_stride as usize) < outer_dim {
        return false;
    }

    if m > blas_index::MAX as usize || n > blas_index::MAX as usize {
        return false;
    }

    true
}

/// Get BLAS compatible layout if any (C or F, preferring the former)
#[cfg(feature = "blas")]
fn get_blas_compatible_layout<S>(a: &ArrayBase<S, Ix2>) -> Option<BlasOrder>
where S: Data
{
    if is_blas_2d(&a.dim, &a.strides, BlasOrder::C) {
        Some(BlasOrder::C)
    } else if is_blas_2d(&a.dim, &a.strides, BlasOrder::F) {
        Some(BlasOrder::F)
    } else {
        None
    }
}

/// `a` should be blas compatible.
/// axis: 0 or 1.
///
/// Return leading stride (lda, ldb, ldc) of array
#[cfg(feature = "blas")]
fn blas_stride<S>(a: &ArrayBase<S, Ix2>, order: BlasOrder) -> blas_index
where S: Data
{
    let axis = order.get_blas_lead_axis();
    let other_axis = 1 - axis;
    let len_this = a.shape()[axis];
    let len_other = a.shape()[other_axis];
    let stride = a.strides()[axis];

    // if current axis has length == 1, then stride does not matter for ndarray
    // but for BLAS we need a stride that makes sense, i.e. it's >= the other axis

    // cast: a should already be blas compatible
    (if len_this <= 1 {
        Ord::max(stride, len_other as isize)
    } else {
        stride
    }) as blas_index
}

#[cfg(test)]
#[cfg(feature = "blas")]
fn blas_row_major_2d<A, S>(a: &ArrayBase<S, Ix2>) -> bool
where
    S: Data,
    A: 'static,
    S::Elem: 'static,
{
    if !same_type::<A, S::Elem>() {
        return false;
    }
    is_blas_2d(&a.dim, &a.strides, BlasOrder::C)
}

#[cfg(test)]
#[cfg(feature = "blas")]
fn blas_column_major_2d<A, S>(a: &ArrayBase<S, Ix2>) -> bool
where
    S: Data,
    A: 'static,
    S::Elem: 'static,
{
    if !same_type::<A, S::Elem>() {
        return false;
    }
    is_blas_2d(&a.dim, &a.strides, BlasOrder::F)
}

#[cfg(test)]
#[cfg(feature = "blas")]
mod blas_tests
{
    use super::*;

    #[test]
    fn blas_row_major_2d_normal_matrix()
    {
        let m: Array2<f32> = Array2::zeros((3, 5));
        assert!(blas_row_major_2d::<f32, _>(&m));
        assert!(!blas_column_major_2d::<f32, _>(&m));
    }

    #[test]
    fn blas_row_major_2d_row_matrix()
    {
        let m: Array2<f32> = Array2::zeros((1, 5));
        assert!(blas_row_major_2d::<f32, _>(&m));
        assert!(blas_column_major_2d::<f32, _>(&m));
    }

    #[test]
    fn blas_row_major_2d_column_matrix()
    {
        let m: Array2<f32> = Array2::zeros((5, 1));
        assert!(blas_row_major_2d::<f32, _>(&m));
        assert!(blas_column_major_2d::<f32, _>(&m));
    }

    #[test]
    fn blas_row_major_2d_transposed_row_matrix()
    {
        let m: Array2<f32> = Array2::zeros((1, 5));
        let m_t = m.t();
        assert!(blas_row_major_2d::<f32, _>(&m_t));
        assert!(blas_column_major_2d::<f32, _>(&m_t));
    }

    #[test]
    fn blas_row_major_2d_transposed_column_matrix()
    {
        let m: Array2<f32> = Array2::zeros((5, 1));
        let m_t = m.t();
        assert!(blas_row_major_2d::<f32, _>(&m_t));
        assert!(blas_column_major_2d::<f32, _>(&m_t));
    }

    #[test]
    fn blas_column_major_2d_normal_matrix()
    {
        let m: Array2<f32> = Array2::zeros((3, 5).f());
        assert!(!blas_row_major_2d::<f32, _>(&m));
        assert!(blas_column_major_2d::<f32, _>(&m));
    }

    #[test]
    fn blas_row_major_2d_skip_rows_ok()
    {
        let m: Array2<f32> = Array2::zeros((5, 5));
        let mv = m.slice(s![..;2, ..]);
        assert!(blas_row_major_2d::<f32, _>(&mv));
        assert!(!blas_column_major_2d::<f32, _>(&mv));
    }

    #[test]
    fn blas_row_major_2d_skip_columns_fail()
    {
        let m: Array2<f32> = Array2::zeros((5, 5));
        let mv = m.slice(s![.., ..;2]);
        assert!(!blas_row_major_2d::<f32, _>(&mv));
        assert!(!blas_column_major_2d::<f32, _>(&mv));
    }

    #[test]
    fn blas_col_major_2d_skip_columns_ok()
    {
        let m: Array2<f32> = Array2::zeros((5, 5).f());
        let mv = m.slice(s![.., ..;2]);
        assert!(blas_column_major_2d::<f32, _>(&mv));
        assert!(!blas_row_major_2d::<f32, _>(&mv));
    }

    #[test]
    fn blas_col_major_2d_skip_rows_fail()
    {
        let m: Array2<f32> = Array2::zeros((5, 5).f());
        let mv = m.slice(s![..;2, ..]);
        assert!(!blas_column_major_2d::<f32, _>(&mv));
        assert!(!blas_row_major_2d::<f32, _>(&mv));
    }

    #[test]
    fn blas_too_short_stride()
    {
        // leading stride must be longer than the other dimension
        // Example, in a 5 x 5 matrix, the leading stride must be >= 5 for BLAS.

        const N: usize = 5;
        const MAXSTRIDE: usize = N + 2;
        let mut data = [0; MAXSTRIDE * N];
        let mut iter = 0..data.len();
        data.fill_with(|| iter.next().unwrap());

        for stride in 1..=MAXSTRIDE {
            let m = ArrayView::from_shape((N, N).strides((stride, 1)), &data).unwrap();
            eprintln!("{:?}", m);

            if stride < N {
                assert_eq!(get_blas_compatible_layout(&m), None);
            } else {
                assert_eq!(get_blas_compatible_layout(&m), Some(BlasOrder::C));
            }
        }
    }
}
