use std::mem;

/// A small map backed by an unsorted vector.
///
/// Maintains key uniqueness at cost of O(n) lookup/insert/remove. Maintains insertion order
/// (`insert` calls that overwrite an existing value don't change order).
#[derive(Clone, Default)]
pub struct VecMap<K, V>(Vec<(K, V)>);

impl<K: Eq, V> VecMap<K, V> {
    pub fn new() -> Self {
        VecMap(Vec::new())
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }

    fn find(&self, key: &K) -> Option<usize> {
        for (i, (k, _)) in self.0.iter().enumerate() {
            if k == key {
                return Some(i);
            }
        }
        None
    }

    pub fn contains(&self, key: &K) -> bool {
        self.find(key).is_some()
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        match self.find(key) {
            Some(i) => Some(&self.0[i].1),
            None => None,
        }
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        match self.find(key) {
            Some(i) => Some(&mut self.0[i].1),
            None => None,
        }
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if let Some(old) = self.get_mut(&key) {
            return Some(mem::replace(old, value));
        }
        self.0.push((key, value));
        None
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        match self.find(key) {
            Some(i) => Some(self.0.remove(i).1),
            None => None,
        }
    }

    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.0.iter().map(|kv| &kv.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty() {
        let m: VecMap<char, u32> = VecMap::new();
        assert!(m.is_empty());
        assert!(!m.contains(&'a'));
        assert!(m.values().next().is_none());
    }

    #[test]
    fn insert_update_get() {
        let mut m: VecMap<char, u32> = VecMap::new();
        assert!(m.insert('a', 1).is_none());
        assert!(m.insert('b', 2).is_none());
        assert_eq!(m.get(&'a'), Some(&1));
        assert_eq!(m.get(&'b'), Some(&2));
        *m.get_mut(&'a').unwrap() += 10;
        assert_eq!(m.get(&'a'), Some(&11));
        assert_eq!(m.get(&'b'), Some(&2));
    }

    #[test]
    fn insert_overwrite() {
        let mut m = VecMap::new();
        assert_eq!(m.insert('a', 1), None);
        assert_eq!(m.insert('b', 2), None);
        assert_eq!(m.insert('a', 3), Some(1));
        assert_eq!(m.insert('a', 4), Some(3));
        assert_eq!(m.get(&'a').copied(), Some(4));
        assert_eq!(m.get(&'b').copied(), Some(2));
        assert_eq!(m.insert('b', 5), Some(2));
        assert_eq!(m.get(&'a').copied(), Some(4));
        assert_eq!(m.get(&'b').copied(), Some(5));
    }

    #[test]
    fn insert_remove() {
        let mut m: VecMap<char, u32> = VecMap::new();
        assert_eq!(m.remove(&'a'), None);
        m.insert('a', 1);
        m.insert('b', 2);
        assert_eq!(m.remove(&'a'), Some(1));
        assert_eq!(m.remove(&'a'), None);
        assert_eq!(m.remove(&'b'), Some(2));
        assert!(m.is_empty());
    }

    #[test]
    fn insertion_order() {
        let mut m: VecMap<char, u32> = VecMap::new();
        let values = |m: &VecMap<char, u32>| -> Vec<u32> { m.values().copied().collect() };
        m.insert('b', 2);
        m.insert('a', 1);
        m.insert('c', 3);
        assert_eq!(values(&m), vec![2, 1, 3]);
        m.insert('a', 11);
        m.remove(&'b');
        assert_eq!(values(&m), vec![11, 3]);
        m.insert('b', 2);
        assert_eq!(values(&m), vec![11, 3, 2]);
    }

    #[test]
    fn containment_equivalences() {
        let mut m = VecMap::new();
        for i in 0u8..=255 {
            if i % 10 < 3 {
                m.insert(i, i);
            }
        }
        for i in 0u8..=255 {
            if m.contains(&i) {
                assert_eq!(m.get(&i).copied(), Some(i));
                assert_eq!(m.get_mut(&i).copied(), Some(i));
                assert_eq!(m.insert(i, i), Some(i));
                assert_eq!(m.remove(&i), Some(i));
            } else {
                assert!(m.get(&i).is_none());
                assert!(m.get_mut(&i).is_none());
                assert!(m.remove(&i).is_none());
                assert!(m.insert(i, i).is_none());
            }
        }
    }

    #[test]
    fn clear() {
        let mut m = VecMap::new();
        m.clear();
        assert!(m.is_empty());
        m.insert('a', 1);
        m.insert('b', 2);
        assert!(!m.is_empty());
        m.clear();
        assert!(m.is_empty());
        m.insert('c', 3);
        assert!(!m.is_empty());
    }
}
