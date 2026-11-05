use crate::dimension::sequence::{Forward, Reverse, Sequence, SequenceMut};
use crate::{Dimension, ErrorKind, Order, ShapeError};

#[inline]
pub(crate) fn reshape_dim<D, E>(from: &D, strides: &D, to: &E, order: Order) -> Result<E, ShapeError>
where
    D: Dimension,
    E: Dimension,
{
    debug_assert_eq!(from.ndim(), strides.ndim());
    let mut to_strides = E::zeros(to.ndim());
    match order {
        Order::RowMajor => {
            reshape_dim_c(&Forward(from), &Forward(strides), &Forward(to), Forward(&mut to_strides))?;
        }
        Order::ColumnMajor => {
            reshape_dim_c(&Reverse(from), &Reverse(strides), &Reverse(to), Reverse(&mut to_strides))?;
        }
    }
    Ok(to_strides)
}

/// Try to reshape an array with dimensions `from_dim` and strides `from_strides` to the new
/// dimension `to_dim`, while keeping the same layout of elements in memory. The strides needed
/// if this is possible are stored into `to_strides`.
///
/// This function uses RowMajor index ordering if the inputs are read in the forward direction
/// (index 0 is axis 0 etc) and ColumnMajor index ordering if the inputs are read in reversed
/// direction (as made possible with the Sequence trait).
///
/// Preconditions:
///
/// 1. from_dim and to_dim are valid dimensions (product of all non-zero axes
///    fits in isize::MAX).
/// 2. from_dim and to_dim are don't have any axes that are zero (that should be handled before
///    this function).
/// 3. `to_strides` should be an all-zeros or all-ones dimension of the right dimensionality
///    (but it will be overwritten after successful exit of this function).
///
/// This function returns:
///
/// - IncompatibleShape if the two shapes are not of matching number of elements
/// - IncompatibleLayout if the input shape and stride can not be remapped to the output shape
///   without moving the array data into a new memory layout.
/// - Ok if the from dim could be mapped to the new to dim.
fn reshape_dim_c<D, E, E2>(from_dim: &D, from_strides: &D, to_dim: &E, mut to_strides: E2) -> Result<(), ShapeError>
where
    D: Sequence<Output = usize>,
    E: Sequence<Output = usize>,
    E2: SequenceMut<Output = usize>,
{
    // cursor indexes into the from and to dimensions
    let mut fi = 0; // index into `from_dim`
    let mut ti = 0; // index into `to_dim`.

    while fi < from_dim.len() && ti < to_dim.len() {
        let mut fd = from_dim[fi];
        let mut fs = from_strides[fi] as isize;
        let mut td = to_dim[ti];

        if fd == td {
            to_strides[ti] = from_strides[fi];
            fi += 1;
            ti += 1;
            continue;
        }

        if fd == 1 {
            fi += 1;
            continue;
        }

        if td == 1 {
            to_strides[ti] = 1;
            ti += 1;
            continue;
        }

        if fd == 0 || td == 0 {
            debug_assert!(false, "zero dim not handled by this function");
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
        }

        // stride times element count is to be distributed out over a combination of axes.
        let mut fstride_whole = fs * (fd as isize);
        let mut fd_product = fd; // cumulative product of axis lengths in the combination (from)
        let mut td_product = td; // cumulative product of axis lengths in the combination (to)

        // The two axis lengths are not a match, so try to combine multiple axes
        // to get it to match up.
        while fd_product != td_product {
            if fd_product < td_product {
                // Take another axis on the from side
                fi += 1;
                if fi >= from_dim.len() {
                    return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
                }
                fd = from_dim[fi];
                fd_product *= fd;
                if fd > 1 {
                    let fs_old = fs;
                    fs = from_strides[fi] as isize;
                    // check if this axis and the next are contiguous together
                    if fs_old != fd as isize * fs {
                        return Err(ShapeError::from_kind(ErrorKind::IncompatibleLayout));
                    }
                }
            } else {
                // Take another axis on the `to` side
                // First assign the stride to the axis we leave behind
                fstride_whole /= td as isize;
                to_strides[ti] = fstride_whole as usize;
                ti += 1;
                if ti >= to_dim.len() {
                    return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
                }

                td = to_dim[ti];
                td_product *= td;
            }
        }

        fstride_whole /= td as isize;
        to_strides[ti] = fstride_whole as usize;

        fi += 1;
        ti += 1;
    }

    // skip past 1-dims at the end
    while fi < from_dim.len() && from_dim[fi] == 1 {
        fi += 1;
    }

    while ti < to_dim.len() && to_dim[ti] == 1 {
        to_strides[ti] = 1;
        ti += 1;
    }

    if fi < from_dim.len() || ti < to_dim.len() {
        return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
    }

    Ok(())
}

#[cfg(feature = "std")]
#[test]
fn test_reshape()
{
    use crate::Dim;

    macro_rules! test_reshape {
        (fail $order:ident from $from:expr, $stride:expr, to $to:expr) => {
            let res = reshape_dim(&Dim($from), &Dim($stride), &Dim($to), Order::$order);
            println!("Reshape {:?} {:?} to {:?}, order {:?}\n  => {:?}",
                     $from, $stride, $to, Order::$order, res);
            let _res = res.expect_err("Expected failed reshape");
        };
        (ok $order:ident from $from:expr, $stride:expr, to $to:expr, $to_stride:expr) => {{
            let res = reshape_dim(&Dim($from), &Dim($stride), &Dim($to), Order::$order);
            println!("Reshape {:?} {:?} to {:?}, order {:?}\n  => {:?}",
                     $from, $stride, $to, Order::$order, res);
            println!("default stride for from dim: {:?}", Dim($from).default_strides());
            println!("default stride for to dim: {:?}", Dim($to).default_strides());
            let res = res.expect("Expected successful reshape");
            assert_eq!(res, Dim($to_stride), "mismatch in strides");
        }};
    }

    test_reshape!(ok C from [1, 2, 3], [6, 3, 1], to [1, 2, 3], [6, 3, 1]);
    test_reshape!(ok C from [1, 2, 3], [6, 3, 1], to [2, 3], [3, 1]);
    test_reshape!(ok C from [1, 2, 3], [6, 3, 1], to [6], [1]);
    test_reshape!(fail C from [1, 2, 3], [6, 3, 1], to [1]);
    test_reshape!(fail F from [1, 2, 3], [6, 3, 1], to [1]);

    test_reshape!(ok C from [6], [1], to [3, 2], [2, 1]);
    test_reshape!(ok C from [3, 4, 5], [20, 5, 1], to [4, 15], [15, 1]);

    test_reshape!(ok C from [4, 4, 4], [16, 4, 1], to [16, 4], [4, 1]);

    test_reshape!(ok C from [4, 4], [4, 1], to [2, 2, 4, 1], [8, 4, 1, 1]);
    test_reshape!(ok C from [4, 4], [4, 1], to [2, 2, 4], [8, 4, 1]);
    test_reshape!(ok C from [4, 4], [4, 1], to [2, 2, 2, 2], [8, 4, 2, 1]);

    test_reshape!(ok C from [4, 4], [4, 1], to [2, 2, 1, 4], [8, 4, 1, 1]);

    test_reshape!(ok C from [4, 4, 4], [16, 4, 1], to [16, 4], [4, 1]);
    test_reshape!(ok C from [3, 4, 4], [16, 4, 1], to [3, 16], [16, 1]);

    test_reshape!(ok C from [4, 4], [8, 1], to [2, 2, 2, 2], [16, 8, 2, 1]);

    test_reshape!(fail C from [4, 4], [8, 1], to [2, 1, 4, 2]);

    test_reshape!(ok C from [16], [4], to [2, 2, 4], [32, 16, 4]);
    test_reshape!(ok C from [16], [-4isize as usize], to [2, 2, 4],
                  [-32isize as usize, -16isize as usize, -4isize as usize]);
    test_reshape!(ok F from [16], [4], to [2, 2, 4], [4, 8, 16]);
    test_reshape!(ok F from [16], [-4isize as usize], to [2, 2, 4],
                  [-4isize as usize, -8isize as usize, -16isize as usize]);

    test_reshape!(ok C from [3, 4, 5], [20, 5, 1], to [12, 5], [5, 1]);
    test_reshape!(ok C from [3, 4, 5], [20, 5, 1], to [4, 15], [15, 1]);
    test_reshape!(fail F from [3, 4, 5], [20, 5, 1], to [4, 15]);
    test_reshape!(ok C from [3, 4, 5, 7], [140, 35, 7, 1], to [28, 15], [15, 1]);

    // preserve stride if shape matches
    test_reshape!(ok C from [10], [2], to [10], [2]);
    test_reshape!(ok F from [10], [2], to [10], [2]);
    test_reshape!(ok C from [2, 10], [1, 2], to [2, 10], [1, 2]);
    test_reshape!(ok F from [2, 10], [1, 2], to [2, 10], [1, 2]);
    test_reshape!(ok C from [3, 4, 5], [20, 5, 1], to [3, 4, 5], [20, 5, 1]);
    test_reshape!(ok F from [3, 4, 5], [20, 5, 1], to [3, 4, 5], [20, 5, 1]);

    test_reshape!(ok C from [3, 4, 5], [4, 1, 1], to [12, 5], [1, 1]);
    test_reshape!(ok F from [3, 4, 5], [1, 3, 12], to [12, 5], [1, 12]);
    test_reshape!(ok F from [3, 4, 5], [1, 3, 1], to [12, 5], [1, 1]);

    // broadcast shapes
    test_reshape!(ok C from [3, 4, 5, 7], [0, 0, 7, 1], to [12, 35], [0, 1]);
    test_reshape!(fail C from [3, 4, 5, 7], [0, 0, 7, 1], to [28, 15]);

    // one-filled shapes
    test_reshape!(ok C from [10], [1], to [1, 10, 1, 1, 1], [1, 1, 1, 1, 1]);
    test_reshape!(ok F from [10], [1], to [1, 10, 1, 1, 1], [1, 1, 1, 1, 1]);
    test_reshape!(ok C from [1, 10], [10, 1], to [1, 10, 1, 1, 1], [10, 1, 1, 1, 1]);
    test_reshape!(ok F from [1, 10], [10, 1], to [1, 10, 1, 1, 1], [10, 1, 1, 1, 1]);
    test_reshape!(ok C from [1, 10], [1, 1], to [1, 5, 1, 1, 2], [1, 2, 2, 2, 1]);
    test_reshape!(ok F from [1, 10], [1, 1], to [1, 5, 1, 1, 2], [1, 1, 5, 5, 5]);
    test_reshape!(ok C from [10, 1, 1, 1, 1], [1, 1, 1, 1, 1], to [10], [1]);
    test_reshape!(ok F from [10, 1, 1, 1, 1], [1, 1, 1, 1, 1], to [10], [1]);
    test_reshape!(ok C from [1, 5, 1, 2, 1], [1, 2, 1, 1, 1], to [10], [1]);
    test_reshape!(fail F from [1, 5, 1, 2, 1], [1, 2, 1, 1, 1], to [10]);
    test_reshape!(ok F from [1, 5, 1, 2, 1], [1, 1, 1, 5, 1], to [10], [1]);
    test_reshape!(fail C from [1, 5, 1, 2, 1], [1, 1, 1, 5, 1], to [10]);
}
