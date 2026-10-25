use std::marker::PhantomData;

use super::Baseiter;
use crate::imp_prelude::*;
use crate::IntoDimension;
use crate::Layout;
use crate::NdProducer;
use crate::Slice;

/// Window producer and iterable
///
/// See [`.windows()`](ArrayBase::windows) for more
/// information.
pub struct Windows<'a, A, D>
{
    base: RawArrayView<A, D>,
    life: PhantomData<&'a A>,
    window: D,
    strides: D,
}

impl<'a, A, D: Dimension> Windows<'a, A, D>
{
    pub(crate) fn new<E>(a: ArrayView<'a, A, D>, window_size: E) -> Self
    where E: IntoDimension<Dim = D>
    {
        let window = window_size.into_dimension();
        let ndim = window.ndim();

        let mut unit_stride = D::zeros(ndim);
        unit_stride.slice_mut().fill(1);

        Windows::new_with_stride(a, window, unit_stride)
    }

    pub(crate) fn new_with_stride<E>(a: ArrayView<'a, A, D>, window_size: E, axis_strides: E) -> Self
    where E: IntoDimension<Dim = D>
    {
        let window = window_size.into_dimension();

        let strides = axis_strides.into_dimension();
        let window_strides = a.strides.clone();

        let base = build_base(a, window.clone(), strides);
        Windows {
            base: base.into_raw_view(),
            life: PhantomData,
            window,
            strides: window_strides,
        }
    }
}

impl_ndproducer! {
    ['a, A, D: Dimension]
    [Clone => 'a, A, D: Clone ]
    Windows {
        base,
        life,
        window,
        strides,
    }
    Windows<'a, A, D> {
        type Item = ArrayView<'a, A, D>;
        type Dim = D;

        unsafe fn item(&self, ptr) {
            ArrayView::new_(ptr, self.window.clone(),
                            self.strides.clone())
        }
    }
}

impl<'a, A, D> IntoIterator for Windows<'a, A, D>
where
    D: Dimension,
    A: 'a,
{
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = WindowsIter<'a, A, D>;
    fn into_iter(self) -> Self::IntoIter
    {
        WindowsIter {
            iter: self.base.into_base_iter(),
            life: self.life,
            window: self.window,
            strides: self.strides,
        }
    }
}

/// Window iterator.
///
/// See [`.windows()`](ArrayBase::windows) for more
/// information.
pub struct WindowsIter<'a, A, D>
{
    iter: Baseiter<A, D>,
    life: PhantomData<&'a A>,
    window: D,
    strides: D,
}

impl_iterator! {
    ['a, A, D: Dimension]
    [Clone => 'a, A, D: Clone]
    WindowsIter {
        iter,
        life,
        window,
        strides,
    }
    WindowsIter<'a, A, D> {
        type Item = ArrayView<'a, A, D>;

        fn item(&mut self, ptr) {
            unsafe {
                ArrayView::new(
                    ptr,
                    self.window.clone(),
                    self.strides.clone())
            }
        }
    }
}

send_sync_read_only!(Windows);
send_sync_read_only!(WindowsIter);

/// Window producer and iterable
///
/// See [`.axis_windows()`](ArrayBase::axis_windows) for more
/// information.
pub struct AxisWindows<'a, A, D>
{
    base: ArrayView<'a, A, D>,
    axis_idx: usize,
    window: D,
    strides: D,
}

impl<'a, A, D: Dimension> AxisWindows<'a, A, D>
{
    pub(crate) fn new(a: ArrayView<'a, A, D>, axis: Axis, window_size: usize) -> Self
    {
        let window_strides = a.strides.clone();
        let axis_idx = axis.index();

        let mut window = a.raw_dim();
        window[axis_idx] = window_size;

        let ndim = window.ndim();
        let mut unit_stride = D::zeros(ndim);
        unit_stride.slice_mut().fill(1);

        let base = build_base(a, window.clone(), unit_stride);
        AxisWindows {
            base,
            axis_idx,
            window,
            strides: window_strides,
        }
    }
}

impl<'a, A, D: Dimension> NdProducer for AxisWindows<'a, A, D>
{
    type Item = ArrayView<'a, A, D>;
    type Dim = Ix1;
    type Ptr = *mut A;
    type Stride = isize;

    fn raw_dim(&self) -> Ix1
    {
        Ix1(self.base.raw_dim()[self.axis_idx])
    }

    fn layout(&self) -> Layout
    {
        self.base.layout()
    }

    fn as_ptr(&self) -> *mut A
    {
        self.base.as_ptr() as *mut _
    }

    fn contiguous_stride(&self) -> isize
    {
        self.base.contiguous_stride()
    }

    unsafe fn as_ref(&self, ptr: *mut A) -> Self::Item
    {
        ArrayView::new_(ptr, self.window.clone(), self.strides.clone())
    }

    unsafe fn uget_ptr(&self, i: &Self::Dim) -> *mut A
    {
        let mut d = D::zeros(self.base.ndim());
        d[self.axis_idx] = i[0];
        self.base.uget_ptr(&d)
    }

    fn stride_of(&self, axis: Axis) -> isize
    {
        assert_eq!(axis, Axis(0));
        self.base.stride_of(Axis(self.axis_idx))
    }

    fn split_at(self, axis: Axis, index: usize) -> (Self, Self)
    {
        assert_eq!(axis, Axis(0));
        let (a, b) = self.base.split_at(Axis(self.axis_idx), index);
        (
            AxisWindows {
                base: a,
                axis_idx: self.axis_idx,
                window: self.window.clone(),
                strides: self.strides.clone(),
            },
            AxisWindows {
                base: b,
                axis_idx: self.axis_idx,
                window: self.window,
                strides: self.strides,
            },
        )
    }

    private_impl!{}
}

impl<'a, A, D> IntoIterator for AxisWindows<'a, A, D>
where
    D: Dimension,
    A: 'a,
{
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = WindowsIter<'a, A, D>;
    fn into_iter(self) -> Self::IntoIter
    {
        WindowsIter {
            iter: self.base.into_base_iter(),
            life: PhantomData,
            window: self.window,
            strides: self.strides,
        }
    }
}

/// build the base array of the `Windows` and `AxisWindows` structs
fn build_base<A, D>(a: ArrayView<A, D>, window: D, strides: D) -> ArrayView<A, D>
where D: Dimension
{
    ndassert!(
        a.ndim() == window.ndim(),
        concat!(
            "Window dimension {} does not match array dimension {} ",
            "(with array of shape {:?})"
        ),
        window.ndim(),
        a.ndim(),
        a.shape()
    );

    ndassert!(
        a.ndim() == strides.ndim(),
        concat!(
            "Stride dimension {} does not match array dimension {} ",
            "(with array of shape {:?})"
        ),
        strides.ndim(),
        a.ndim(),
        a.shape()
    );

    let mut base = a;
    base.slice_each_axis_inplace(|ax_desc| {
        let len = ax_desc.len;
        let wsz = window[ax_desc.axis.index()];
        let stride = strides[ax_desc.axis.index()];

        if len < wsz {
            Slice::new(0, Some(0), 1)
        } else {
            Slice::new(0, Some((len - wsz + 1) as isize), stride as isize)
        }
    });
    base
}
