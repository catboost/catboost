#[repr(C)]
struct Wrapper<T, const Y: usize, const Z: usize>([T; Y], [T; Z]);

const fn check_impl<T, const X: usize, const Y: usize, const Z: usize>() {
    // Check that the array sizes match
    assert!(X == Y + Z, "Array cannot be split: sizes don't match");
    assert!(
        usize::checked_add(Y, Z).is_some(),
        "Array cannot be split: length would overflow"
    );
    // Make doubly sure that nothing funky is going on with the memory representations
    assert!(core::mem::size_of::<Wrapper<T, Y, Z>>() == core::mem::size_of::<[T; X]>());
    assert!(core::mem::align_of::<Wrapper<T, Y, Z>>() == core::mem::align_of::<[T; X]>());
}

trait SizeCheck<T, const X: usize, const Y: usize, const Z: usize> {
    const CHECK: Self;
}

impl<T, const X: usize, const Y: usize, const Z: usize> SizeCheck<T, X, Y, Z> for () {
    const CHECK: () = check_impl::<T, X, Y, Z>();
}

#[allow(clippy::let_unit_value)]
fn check<T, const X: usize, const Y: usize, const Z: usize>() {
    let _: () = SizeCheck::<T, X, Y, Z>::CHECK;
    // Do the same checks at run-time, so even if rustc changes to allow ignore
    // our compile-time errors, we will at least not create UB
    check_impl::<T, X, Y, Z>();
}

pub(crate) fn split_mut<'a, T, const X: usize, const Y: usize, const Z: usize>(
    x: &'a mut [T; X],
) -> (&'a mut [T; Y], &'a mut [T; Z]) {
    check::<T, X, Y, Z>();

    let wrapper: &'a mut Wrapper<T, Y, Z> = unsafe { core::mem::transmute(x) };
    (&mut wrapper.0, &mut wrapper.1)
}
