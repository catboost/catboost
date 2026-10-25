use core::sync::atomic::Ordering;

use bytemuck::NoUninit;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::Atomic;

impl<T> Serialize for Atomic<T>
where
    T: NoUninit + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Matches the atomic ordering used in `Debug` for `Atomic<T>`.
        self.load(Ordering::Relaxed).serialize(serializer)
    }
}

impl<'de, T> Deserialize<'de> for Atomic<T>
where
    T: for<'a> Deserialize<'a>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Deserialize::deserialize(deserializer).map(Self::new)
    }
}
