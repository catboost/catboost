mod se {
    use crate::SizedHashMap;
    use core::hash::Hash;
    use serde::ser::{Serialize, SerializeMap, Serializer};

    impl<K, V, H, const N: usize> Serialize for SizedHashMap<K, V, H, N>
    where
        K: Eq + Hash + Serialize,
        V: Serialize,
    {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let mut map = serializer.serialize_map(Some(self.len()))?;
            for (k, v) in self.iter() {
                map.serialize_entry(k, v)?;
            }
            map.end()
        }
    }
}

mod de {
    use crate::SizedHashMap;
    use core::hash::Hash;
    use core::marker::PhantomData;
    use serde::de::{Deserialize, Deserializer, MapAccess, Visitor};
    use std::fmt;
    use std::hash::BuildHasher;

    mod size_hint {
        use core::cmp;

        /// This presumably exists to prevent denial of service attacks.
        ///
        /// Original discussion: <https://github.com/serde-rs/serde/issues/1114>.
        #[inline]
        pub(super) fn cautious(hint: Option<usize>) -> usize {
            cmp::min(hint.unwrap_or(0), 4096)
        }
    }

    impl<'de, K, V, H, const N: usize> Deserialize<'de> for SizedHashMap<K, V, H, N>
    where
        K: Eq + Hash + Deserialize<'de>,
        V: Deserialize<'de>,
        H: Default + BuildHasher,
    {
        fn deserialize<D>(deserializer: D) -> Result<SizedHashMap<K, V, H, N>, D::Error>
        where
            D: Deserializer<'de>,
        {
            deserializer.deserialize_any(HashMapVisitor {
                marker: PhantomData,
            })
        }
    }

    struct HashMapVisitor<K, V, H, const N: usize>
    where
        K: Eq + Hash,
        H: Default + BuildHasher,
    {
        marker: PhantomData<SizedHashMap<K, V, H, N>>,
    }

    impl<'de, K, V, H, const N: usize> Visitor<'de> for HashMapVisitor<K, V, H, N>
    where
        K: Eq + Hash + Deserialize<'de>,
        V: Deserialize<'de>,
        H: Default + BuildHasher,
    {
        type Value = SizedHashMap<K, V, H, N>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("an Object/Map structure")
        }

        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: MapAccess<'de>,
        {
            let size = size_hint::cautious(map.size_hint());

            let mut m = SizedHashMap::with_capacity_and_hasher(size, H::default());
            while let Some(k) = map.next_key()? {
                let v = map.next_value()?;
                m.insert(k, v);
            }
            Ok(m)
        }
    }
}
