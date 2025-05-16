use core::{
    hash::BuildHasherDefault,
    ops::{Deref, DerefMut},
};
use std::collections::{BinaryHeap, HashMap};

use crate::{Analyzation, HashLocation, LocationHasher};

#[derive(Debug, Default)]
pub struct ForkData {
    pub data:
        HashMap<HashLocation<'static>, BinaryHeap<Analyzation>, BuildHasherDefault<LocationHasher>>,
}

impl ForkData {
    #[inline]
    pub fn new() -> ForkData {
        ForkData::default()
    }
}

impl Deref for ForkData {
    type Target =
        HashMap<HashLocation<'static>, BinaryHeap<Analyzation>, BuildHasherDefault<LocationHasher>>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for ForkData {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

#[cfg(feature = "serde")]
mod serde {
    use std::collections::HashMap;

    use serde::{
        Deserialize, Serialize,
        de::{MapAccess, Visitor},
        ser::SerializeMap,
    };

    use crate::HashLocation;

    use super::ForkData;

    impl Serialize for ForkData {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            // serde/serde_json does not automatically convert customs key struct in a map to a string:
            let mut map = serializer.serialize_map(Some(self.data.len()))?;
            for (k, v) in self.data.iter() {
                map.serialize_entry(&k.to_string(), &v)?;
            }
            map.end()
        }
    }

    pub struct ForkDataVisitor;

    impl Visitor<'static> for ForkDataVisitor {
        type Value = ForkData;

        fn expecting(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
            formatter.write_str("wrapper for map")
        }

        fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
        where
            M: MapAccess<'static>,
        {
            let mut data: HashMap<&str, _> = HashMap::with_capacity_and_hasher(
                access.size_hint().unwrap_or(0),
                Default::default(),
            );

            while let Some((key, value)) = access.next_entry()? {
                data.insert(key, value);
            }

            let data = data
                .into_iter()
                .map(|(key, value)| {
                    let mut key_split = key.split(',');
                    let file = key_split.next().unwrap();
                    let line = key_split.next().unwrap().parse().unwrap();
                    let col = key_split.next().unwrap().parse().unwrap();
                    (HashLocation { file, line, col }, value)
                })
                .collect();

            Ok(ForkData { data })
        }
    }

    impl Deserialize<'static> for ForkData {
        #[inline]
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'static>,
        {
            deserializer.deserialize_map(ForkDataVisitor {})
        }
    }
}
