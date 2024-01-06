use core::cell::RefCell;

use serde::{ser::SerializeStruct,
    Deserialize, Deserializer, Serialize, Serializer,
};

use crate::VERSION;

use super::{Fork, fork_data::ForkData};

impl<Mods> Serialize for Fork<Mods> {
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let data = self.gpu_or_cpu.borrow();

        let mut state = serializer.serialize_struct("Fork", 2)?;
        state.serialize_field("version", VERSION)?;
        state.serialize_field("gpu_or_cpu", &*data)?;
        state.end()
    }
}


impl<Mods> Fork<Mods> {
    #[inline]
    pub fn load_from_deserializer<D: Deserializer<'static>>(
        &mut self,
        deserializer: D,
    ) -> Result<(), D::Error> {
        self.gpu_or_cpu = RefCell::new(ForkData::deserialize(deserializer)?);
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

        use crate::{ApplyFunction, Base, Cached, Combiner, Device, Fork, OpenCL};

        let mut device = OpenCL::<Fork<Cached<Base>>>::new(0).unwrap();
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

        let mut json = vec![];
        let mut serializer = serde_json::Serializer::new(&mut json);

        drop(buf);
        drop(out);

        // let map = HashMap::from([((32, 32), 53)]);

        device.modules.serialize(&mut serializer).unwrap();

        let json = Box::leak(json.into_boxed_slice());

        let mut de = serde_json::Deserializer::from_slice(json);

        device.modules.load_from_deserializer(&mut de).unwrap();
        // let json = String::from_utf8(json).unwrap();



        // println!("json: {json:?}");
        // device.modules.save_as_json(".")
    }
}
