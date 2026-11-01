#[cfg(not(feature = "arraybackend"))]
pub(crate) type VecDrain<'a, T, const N: usize> = std::vec::Drain<'a, T>;
#[cfg(feature = "arraybackend")]
pub(crate) type VecDrain<'a, T, const N: usize> = arrayvec::Drain<'a, T, N>;

#[cfg(not(feature = "arraybackend"))]
pub(crate) type VecIntoIter<T, const N: usize> = std::vec::IntoIter<T>;
#[cfg(feature = "arraybackend")]
pub(crate) type VecIntoIter<T, const N: usize> = arrayvec::IntoIter<T, N>;
