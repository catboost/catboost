use super::VecMap;
use crate::vectypes::VecIntoIter;

impl<K, V, const N: usize, S> IntoIterator for VecMap<K, V, N, S> {
    type Item = (K, V);
    type IntoIter = VecIntoIter<(K, V), N>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.v.into_iter()
    }
}
