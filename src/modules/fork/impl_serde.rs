use serde::{Deserialize, Deserializer};

use super::Fork;
use crate::Base;

// impl Deserialize<'static> for Fork<Base> {
//     fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
//     where
//         D: Deserializer<'static>
//     {
//         deserializer.deserialize_struct("Fork", &["version", "gpu_or_cpu"], )
//     }
// }

impl<Mods> Fork<Mods> {
    #[inline]
    pub fn load_from_deserializer<D: Deserializer<'static>>(
        &mut self,
        deserializer: D,
    ) -> Result<(), D::Error> {
        let de_fork = Fork::<Base>::deserialize(deserializer)?;
        self.gpu_or_cpu = de_fork.gpu_or_cpu;
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
        use serde::Serialize;

        use crate::{ApplyFunction, Base, Cached, Combiner, Device, Fork, OpenCL};

        let mut device = OpenCL::<Fork<Cached<Base>>>::new(0).unwrap();
        // let data_prev = device.modules.gpu_or_cpu.borrow().clone();

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
        // let data_now = device.modules.gpu_or_cpu.borrow().clone();
        // println!("data_now: {data_now:?}");
        // assert_eq!(data_now, data_prev);
    }
}
