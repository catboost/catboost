use crate::dimension::IntoDimension;
use crate::order::Order;
use crate::Dimension;

/// A contiguous array shape of n dimensions.
///
/// Either c- or f- memory ordered (*c* a.k.a *row major* is the default).
#[derive(Copy, Clone, Debug)]
pub struct Shape<D>
{
    /// Shape (axis lengths)
    pub(crate) dim: D,
    /// Strides can only be C or F here
    pub(crate) strides: Strides<Contiguous>,
}

#[derive(Copy, Clone, Debug)]
pub(crate) enum Contiguous {}

impl<D> Shape<D>
{
    pub(crate) fn is_c(&self) -> bool
    {
        matches!(self.strides, Strides::C)
    }
}

/// An array shape of n dimensions in c-order, f-order or custom strides.
#[derive(Copy, Clone, Debug)]
pub struct StrideShape<D>
{
    pub(crate) dim: D,
    pub(crate) strides: Strides<D>,
}

impl<D> StrideShape<D>
where D: Dimension
{
    /// Return a reference to the dimension
    pub fn raw_dim(&self) -> &D
    {
        &self.dim
    }
    /// Return the size of the shape in number of elements
    pub fn size(&self) -> usize
    {
        self.dim.size()
    }
}

/// Stride description
#[derive(Copy, Clone, Debug)]
pub(crate) enum Strides<D>
{
    /// Row-major ("C"-order)
    C,
    /// Column-major ("F"-order)
    F,
    /// Custom strides
    Custom(D),
}

impl<D> Strides<D>
{
    /// Return strides for `dim` (computed from dimension if c/f, else return the custom stride)
    pub(crate) fn strides_for_dim(self, dim: &D) -> D
    where D: Dimension
    {
        match self {
            Strides::C => dim.default_strides(),
            Strides::F => dim.fortran_strides(),
            Strides::Custom(c) => {
                debug_assert_eq!(
                    c.ndim(),
                    dim.ndim(),
                    "Custom strides given with {} dimensions, expected {}",
                    c.ndim(),
                    dim.ndim()
                );
                c
            }
        }
    }

    #[inline]
    pub(crate) fn is_custom(&self) -> bool
    {
        matches!(*self, Strides::Custom(_))
    }
}

/// A trait for `Shape` and `D where D: Dimension` that allows
/// customizing the memory layout (strides) of an array shape.
///
/// This trait is used together with array constructor methods like
/// `Array::from_shape_vec`.
pub trait ShapeBuilder
{
    type Dim: Dimension;
    type Strides;

    fn into_shape_with_order(self) -> Shape<Self::Dim>;
    fn f(self) -> Shape<Self::Dim>;
    fn set_f(self, is_f: bool) -> Shape<Self::Dim>;
    fn strides(self, strides: Self::Strides) -> StrideShape<Self::Dim>;
}

impl<D> From<D> for Shape<D>
where D: Dimension
{
    /// Create a `Shape` from `dimension`, using the default memory layout.
    fn from(dimension: D) -> Shape<D>
    {
        dimension.into_shape_with_order()
    }
}

impl<T, D> From<T> for StrideShape<D>
where
    D: Dimension,
    T: ShapeBuilder<Dim = D>,
{
    fn from(value: T) -> Self
    {
        let shape = value.into_shape_with_order();
        let st = if shape.is_c() { Strides::C } else { Strides::F };
        StrideShape {
            strides: st,
            dim: shape.dim,
        }
    }
}

impl<T> ShapeBuilder for T
where T: IntoDimension
{
    type Dim = T::Dim;
    type Strides = T;
    fn into_shape_with_order(self) -> Shape<Self::Dim>
    {
        Shape {
            dim: self.into_dimension(),
            strides: Strides::C,
        }
    }
    fn f(self) -> Shape<Self::Dim>
    {
        self.set_f(true)
    }
    fn set_f(self, is_f: bool) -> Shape<Self::Dim>
    {
        self.into_shape_with_order().set_f(is_f)
    }
    fn strides(self, st: T) -> StrideShape<Self::Dim>
    {
        self.into_shape_with_order().strides(st.into_dimension())
    }
}

impl<D> ShapeBuilder for Shape<D>
where D: Dimension
{
    type Dim = D;
    type Strides = D;

    fn into_shape_with_order(self) -> Shape<D>
    {
        self
    }

    fn f(self) -> Self
    {
        self.set_f(true)
    }

    fn set_f(mut self, is_f: bool) -> Self
    {
        self.strides = if !is_f { Strides::C } else { Strides::F };
        self
    }

    fn strides(self, st: D) -> StrideShape<D>
    {
        StrideShape {
            dim: self.dim,
            strides: Strides::Custom(st),
        }
    }
}

impl<D> Shape<D>
where D: Dimension
{
    /// Return a reference to the dimension
    pub fn raw_dim(&self) -> &D
    {
        &self.dim
    }
    /// Return the size of the shape in number of elements
    pub fn size(&self) -> usize
    {
        self.dim.size()
    }
}

/// Array shape argument with optional order parameter
///
/// Shape or array dimension argument, with optional [`Order`] parameter.
///
/// This is an argument conversion trait that is used to accept an array shape and
/// (optionally) an ordering argument.
///
/// See for example [`.to_shape()`](crate::ArrayBase::to_shape).
pub trait ShapeArg
{
    type Dim: Dimension;
    fn into_shape_and_order(self) -> (Self::Dim, Option<Order>);
}

impl<T> ShapeArg for T
where T: IntoDimension
{
    type Dim = T::Dim;

    fn into_shape_and_order(self) -> (Self::Dim, Option<Order>)
    {
        (self.into_dimension(), None)
    }
}

impl<T> ShapeArg for (T, Order)
where T: IntoDimension
{
    type Dim = T::Dim;

    fn into_shape_and_order(self) -> (Self::Dim, Option<Order>)
    {
        (self.0.into_dimension(), Some(self.1))
    }
}
