use core::cell::RefCell;
use std::collections::HashMap;
use serde::ser::SerializeMap;

use super::Fork;

#[cfg(feature = "serde")]
impl<Mods> serde::Serialize for Fork<Mods> {
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer 
    {
        let data = self.gpu_or_cpu.borrow();

        // serde/serde_json does not automatically convert customs key struct in a map to a string:
        let mut map = serializer.serialize_map(Some(data.len()))?;
        for (k, v) in data.iter() {
            map.serialize_entry(&k.to_string(), &v)?;
        }
        map.end()
    }
}

#[cfg(all(feature = "serde", not(feature = "no-std")))]
impl<Mods> Fork<Mods> {
    #[inline]
    pub fn load_from_deserializer<D: serde::Deserializer<'static>>(&mut self, deserializer: D) -> Result<(), D::Error> {
        use serde::Deserialize;

        self.gpu_or_cpu = RefCell::new(HashMap::deserialize(deserializer)?);
        Ok(())
    }

    #[cfg(feature = "json")]
    #[inline]
    pub fn save_as_json(&self, path: impl AsRef<std::path::Path>) -> crate::Result<()> {
        serde_json::to_writer(std::fs::File::open(path)?, self)?;
        Ok(())
    }

    #[cfg(feature = "json")]
    #[inline]
    pub fn load_from_json_read(&mut self, reader: impl std::io::Read) -> serde_json::Result<()> {
        self.load_from_deserializer(&mut serde_json::Deserializer::from_reader(reader))
    }
    
    #[cfg(feature = "json")]
    #[inline]
    pub fn load_from_json(&mut self, path: impl AsRef<std::path::Path>) -> crate::Result<()> {
        self.load_from_json_read(std::fs::File::open(path)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "json")]
    #[cfg(feature = "serde")]
    #[cfg(feature = "opencl")]
    #[test]
    fn test_fork_deserialize() {
        use std::collections::HashMap;

        use serde::Serialize;

        use crate::{OpenCL, Fork, Cached, Base, ApplyFunction, Device, Combiner};

        let device = OpenCL::<Fork<Cached<Base>>>::new(0).unwrap();
        if !device.unified_mem() {
            return;
        }
        let buf = device.buffer([1, 2, 4, 5, 6, 7]);
        let out = device.apply_fn(&buf, |x| x.add(3));
        assert_eq!(out.read(), [4, 5, 7, 8, 9, 10]);

        for _ in 0..100 {
            let _out = device.apply_fn(&buf, |x| x.add(3));
            let gpu_or_cpu = device.modules.gpu_or_cpu.borrow();
            let (_, operations) = gpu_or_cpu.iter().next().unwrap();
            assert_eq!(operations.len(), 2);
            let analyzations = operations.iter().cloned().collect::<Vec<_>>();
            assert_eq!(&analyzations[0].input_lengths, &[6]);
        }

        let mut json = vec![0,];
        let mut serializer = serde_json::Serializer::new(&mut json);

        let map = HashMap::from([((32, 32), 53)]);

        map.serialize(&mut serializer).unwrap();
        println!("json: {json:?}");
        // device.modules.save_as_json(".")
    }

   
}