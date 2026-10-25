use crate::imp_prelude::*;

/// Adds the two dimensions at compile time.
pub trait DimAdd<D: Dimension>
{
    /// The sum of the two dimensions.
    type Output: Dimension;
}

macro_rules! impl_dimadd_const_out_const {
    ($lhs:expr, $rhs:expr) => {
        impl DimAdd<Dim<[usize; $rhs]>> for Dim<[usize; $lhs]> {
            type Output = Dim<[usize; $lhs + $rhs]>;
        }
    };
}

macro_rules! impl_dimadd_const_out_dyn {
    ($lhs:expr, IxDyn) => {
        impl DimAdd<IxDyn> for Dim<[usize; $lhs]> {
            type Output = IxDyn;
        }
    };
    ($lhs:expr, $rhs:expr) => {
        impl DimAdd<Dim<[usize; $rhs]>> for Dim<[usize; $lhs]> {
            type Output = IxDyn;
        }
    };
}

impl<D: Dimension> DimAdd<D> for Ix0
{
    type Output = D;
}

impl_dimadd_const_out_const!(1, 0);
impl_dimadd_const_out_const!(1, 1);
impl_dimadd_const_out_const!(1, 2);
impl_dimadd_const_out_const!(1, 3);
impl_dimadd_const_out_const!(1, 4);
impl_dimadd_const_out_const!(1, 5);
impl_dimadd_const_out_dyn!(1, 6);
impl_dimadd_const_out_dyn!(1, IxDyn);

impl_dimadd_const_out_const!(2, 0);
impl_dimadd_const_out_const!(2, 1);
impl_dimadd_const_out_const!(2, 2);
impl_dimadd_const_out_const!(2, 3);
impl_dimadd_const_out_const!(2, 4);
impl_dimadd_const_out_dyn!(2, 5);
impl_dimadd_const_out_dyn!(2, 6);
impl_dimadd_const_out_dyn!(2, IxDyn);

impl_dimadd_const_out_const!(3, 0);
impl_dimadd_const_out_const!(3, 1);
impl_dimadd_const_out_const!(3, 2);
impl_dimadd_const_out_const!(3, 3);
impl_dimadd_const_out_dyn!(3, 4);
impl_dimadd_const_out_dyn!(3, 5);
impl_dimadd_const_out_dyn!(3, 6);
impl_dimadd_const_out_dyn!(3, IxDyn);

impl_dimadd_const_out_const!(4, 0);
impl_dimadd_const_out_const!(4, 1);
impl_dimadd_const_out_const!(4, 2);
impl_dimadd_const_out_dyn!(4, 3);
impl_dimadd_const_out_dyn!(4, 4);
impl_dimadd_const_out_dyn!(4, 5);
impl_dimadd_const_out_dyn!(4, 6);
impl_dimadd_const_out_dyn!(4, IxDyn);

impl_dimadd_const_out_const!(5, 0);
impl_dimadd_const_out_const!(5, 1);
impl_dimadd_const_out_dyn!(5, 2);
impl_dimadd_const_out_dyn!(5, 3);
impl_dimadd_const_out_dyn!(5, 4);
impl_dimadd_const_out_dyn!(5, 5);
impl_dimadd_const_out_dyn!(5, 6);
impl_dimadd_const_out_dyn!(5, IxDyn);

impl_dimadd_const_out_const!(6, 0);
impl_dimadd_const_out_dyn!(6, 1);
impl_dimadd_const_out_dyn!(6, 2);
impl_dimadd_const_out_dyn!(6, 3);
impl_dimadd_const_out_dyn!(6, 4);
impl_dimadd_const_out_dyn!(6, 5);
impl_dimadd_const_out_dyn!(6, 6);
impl_dimadd_const_out_dyn!(6, IxDyn);

impl<D: Dimension> DimAdd<D> for IxDyn
{
    type Output = IxDyn;
}
