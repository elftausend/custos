use core::{any::Any, hash::BuildHasherDefault};
use std::{collections::HashMap, ffi::c_void, sync::Arc};

use crate::{AllocFlag, DeviceError, Unit};

use super::CLPtr;
use crate::{
    Base, Buffer, CachedCPU, CachedModule, Cursor, Device, OnDropBuffer, OpenCL, Shape,
    UnifiedMemChain, UniqueId, CPU,
};
use min_cl::api::{create_buffer, MemFlags};

impl<Mods: UnifiedMemChain<Self> + OnDropBuffer> UnifiedMemChain<Self> for OpenCL<Mods> {
    #[inline]
    fn construct_unified_buf_from_cpu_buf<'a, T: Unit + 'static, S: Shape>(
        &self,
        device: &'a Self,
        no_drop_buf: Buffer<'a, T, CachedCPU, S>,
    ) -> crate::Result<Buffer<'a, T, Self, S>> {
        self.modules
            .construct_unified_buf_from_cpu_buf(device, no_drop_buf)
    }
}

impl<Mods, OclMods, SimpleMods> UnifiedMemChain<OpenCL<OclMods>>
for CachedModule<Mods, OpenCL<SimpleMods>>
    where
        OclMods: Cursor + OnDropBuffer,
        SimpleMods: OnDropBuffer,
{
    #[inline]
    fn construct_unified_buf_from_cpu_buf<'a, T: Unit + 'static, S: Shape>(
        &self,
        device: &'a OpenCL<OclMods>,
        no_drop_buf: Buffer<'a, T, CachedCPU, S>,
    ) -> crate::Result<Buffer<'a, T, OpenCL<OclMods>, S>> {
        construct_buffer(
            device,
            no_drop_buf,
            &mut self.cache.borrow_mut().nodes, // Pass the cache directly
            device.cursor() as UniqueId,
        )
    }
}

impl<D: Device> UnifiedMemChain<D> for Base {
    #[inline]
    fn construct_unified_buf_from_cpu_buf<'a, T: Unit, S: Shape>(
        &self,
        _device: &'a D,
        _no_drop_buf: Buffer<'a, T, CachedCPU, S>,
    ) -> crate::Result<Buffer<'a, T, D, S>> {
        Err(DeviceError::UnifiedConstructNotAvailable.into())
    }
}

pub unsafe fn to_cached_unified<OclMods, CpuMods, T, S>(
    device: &OpenCL<OclMods>,
    no_drop: Buffer<T, CPU<CpuMods>, S>,
    cache: &mut HashMap<(u64, usize), Arc<dyn Any>, BuildHasherDefault<crate::NoHasher>>,
    key: (u64, usize),
) -> crate::Result<*mut c_void>
    where
        OclMods: OnDropBuffer,
        CpuMods: OnDropBuffer,
        T: Unit + 'static,
        S: Shape,
{
    let cl_ptr = create_buffer(
        device.ctx(),
        MemFlags::MemReadWrite | MemFlags::MemUseHostPtr,
        no_drop.len(),
        Some(&no_drop),
    )?;

    let old_ptr = cache.insert(
        key,
        Arc::new(CLPtr {
            ptr: cl_ptr,
            host_ptr: no_drop.base().ptr,
            len: no_drop.len(),
            flag: AllocFlag::None,
        }),
    );

    drop(old_ptr);

    Ok(cl_ptr)
}

pub fn construct_buffer<'a, OclMods, CpuMods, T, S>(
    device: &'a OpenCL<OclMods>,
    no_drop: Buffer<'a, T, CPU<CpuMods>, S>,
    cache: &mut HashMap<(u64, usize), Arc<dyn Any>, BuildHasherDefault<crate::NoHasher>>,
    id: crate::UniqueId,
) -> crate::Result<Buffer<'a, T, OpenCL<OclMods>, S>>
    where
        OclMods: Cursor + OnDropBuffer,
        CpuMods: OnDropBuffer,
        T: Unit + 'static,
        S: Shape,
{
    use crate::PtrType;

    if no_drop.data.flag() == AllocFlag::None {
        return Err(DeviceError::UnifiedConstructInvalidInputBuffer.into());
    }

    unsafe { device.bump_cursor() };

    let key = (id, no_drop.len());

    if let Some(rawcl) = cache.get(&key) {
        let rawcl = rawcl
            .downcast_ref::<<OpenCL<OclMods> as Device>::Base<T, S>>()
            .unwrap();
        let data = device.base_to_data::<T, S>(CLPtr {
            ptr: rawcl.ptr,
            host_ptr: rawcl.host_ptr,
            len: no_drop.len(),
            flag: no_drop.data.flag(),
        });
        return Ok(Buffer {
            data,
            device: Some(device),
        });
    }

    let (host_ptr, len) = (no_drop.base().ptr, no_drop.len());
    let ptr = unsafe { to_cached_unified(device, no_drop, cache, key)? };

    let data = device.base_to_data::<T, S>(CLPtr {
        ptr,
        host_ptr,
        len,
        flag: AllocFlag::Wrapper,
    });

    Ok(Buffer {
        data,
        device: Some(device),
    })
}

#[cfg(unified_cl)]
#[cfg(test)]
mod tests {
    use crate::{
        opencl::{chosen_cl_idx, CLPtr},
        AllocFlag, Base, Buffer, Cache, Cached, Device, DeviceError, OpenCL, Retriever,
        UnifiedMemChain, CPU,
    };

    use super::{construct_buffer, to_cached_unified};

    #[test]
    fn test_unified_mem_chain_ideal() -> crate::Result<()> {
        let cpu = CPU::<Cached<Base>>::new();
        let mut no_drop = cpu.retrieve::<0>(3, ()).unwrap();
        no_drop.write(&[1., 2.3, 0.76]);

        let device = OpenCL::<Cached<Base>>::new(0)?;
        let buf = device.construct_unified_buf_from_cpu_buf::<_, ()>(&device, no_drop)?;
        assert_eq!(buf.read(), &[1., 2.3, 0.76]);
        Ok(())
    }

    #[test]
    fn test_unified_mem_chain_ideal_using_cpu_from_opencl_dev() -> crate::Result<()> {
        let device = OpenCL::<Cached<Base>>::new(0)?;

        let mut no_drop = device.cpu.retrieve::<0>(3, ()).unwrap();
        no_drop.write(&[1., 2.3, 0.76]);

        let buf = device.construct_unified_buf_from_cpu_buf::<_, ()>(&device, no_drop)?;

        assert_eq!(buf.read(), &[1., 2.3, 0.76]);
        Ok(())
    }

    #[test]
    fn test_unified_mem_chain_invalid_no_drop_buf() -> crate::Result<()> {
        let cpu = CPU::<Cached<Base>>::new();
        let no_drop = cpu.buffer([1, 2, 3]);

        let device = OpenCL::<Cached<Base>>::new(0)?;
        let buf = device.construct_unified_buf_from_cpu_buf(&device, no_drop);
        match buf
            .expect_err("Missing error -> failure")
            .downcast_ref::<DeviceError>()
            .unwrap()
        {
            DeviceError::UnifiedConstructInvalidInputBuffer => Ok(()),
            _ => panic!("wrong error"),
        }
    }

    #[test]
    fn test_unified_mem_chain_unified_construct_unavailable() -> crate::Result<()> {
        let cpu = CPU::<Cached<Base>>::new();
        let mut no_drop: Buffer<f64, _> = cpu.retrieve::<0>(3, ()).unwrap();
        no_drop.write(&[1., 2.3, 0.76]);

        let device = OpenCL::<Base>::new(chosen_cl_idx())?;
        let buf = device.construct_unified_buf_from_cpu_buf(&device, no_drop);
        match buf
            .expect_err("Missing error -> failure")
            .downcast_ref::<DeviceError>()
            .unwrap()
        {
            DeviceError::UnifiedConstructNotAvailable => Ok(()),
            _ => panic!("wrong error"),
        }
    }

    #[test]
    fn test_construct_buffer_missing_cached_module() -> crate::Result<()> {
        let cpu = CPU::<Base>::new();
        let mut no_drop: Buffer<f64, _> = cpu.retrieve::<0>(3, ()).unwrap();
        no_drop.write(&[1., 2.3, 0.76]);

        let device = OpenCL::<Base>::new(chosen_cl_idx())?;
        let mut cache = Cache::new();

        let buf = construct_buffer(&device, no_drop, &mut cache.nodes, 0);
        match buf
            .expect_err("Missing error -> failure")
            .downcast_ref::<DeviceError>()
            .unwrap()
        {
            DeviceError::UnifiedConstructInvalidInputBuffer => Ok(()),
            _ => panic!("wrong error"),
        }
    }

    #[test]
    fn test_to_unified() -> crate::Result<()> {
        let cpu = CPU::<Cached<Base>>::new();
        let mut no_drop: Buffer<_, _> = cpu.retrieve::<0>(3, ()).unwrap();
        no_drop.write(&[1., 2.3, 0.76]);

        let device = OpenCL::<Base>::new(chosen_cl_idx())?;
        let mut cache = Cache::new();

        let (host_ptr, len) = (no_drop.data.ptr, no_drop.len());
        let cl_host_ptr = unsafe { to_cached_unified(&device, no_drop, &mut cache.nodes, (0, 0))? };

        let buf: Buffer<f32, OpenCL> = Buffer {
            data: CLPtr {
                ptr: cl_host_ptr,
                host_ptr,
                len,
                flag: AllocFlag::Wrapper,
            },
            device: Some(&device),
        };

        assert_eq!(buf.read(), vec![1., 2.3, 0.76]);
        assert_eq!(buf.read(), &[1., 2.3, 0.76]);
        Ok(())
    }

    #[test]
    fn test_construct_buffer() -> crate::Result<()> {
        let cpu = CPU::<Cached<Base>>::new();
        let mut no_drop = cpu.retrieve::<0>(3, ()).unwrap();
        no_drop.write(&[1., 2.3, 0.76]);

        let device = OpenCL::<Base>::new(chosen_cl_idx())?;
        let mut cache = Cache::new();

        let buf: Buffer<_, _> = construct_buffer(&device, no_drop, &mut cache.nodes, 0)?;

        assert_eq!(buf.read(), vec![1., 2.3, 0.76]);
        assert_eq!(buf.read(), &[1., 2.3, 0.76]);

        Ok(())
    }

    #[cfg(unified_cl)]
    #[test]
    fn test_cpu_to_unified_is_reusing_converted_buf() -> crate::Result<()> {
        use crate::{Base, Cached, Cursor, Retriever, UniqueId};
        use std::time::Instant;

        let cl_dev = OpenCL::<Cached<Base>>::new(0)?;
        let device = CPU::<Cached<Base>>::new();

        let mut dur = 0.;

        for _ in cl_dev.range(0..100) {
            let mut buf: Buffer<i32, _> = device.retrieve::<0>(6, ()).unwrap();

            buf.copy_from_slice(&[1, 2, 3, 4, 5, 6]);

            let start = Instant::now();
            let cl_cpu_buf = construct_buffer(
                &cl_dev,
                buf,
                &mut cl_dev.modules.cache.borrow_mut().nodes,
                cl_dev.cursor() as UniqueId,
            )?;
            dur += start.elapsed().as_secs_f64();

            assert_eq!(cl_cpu_buf.read(), &[1, 2, 3, 4, 5, 6]);
            assert_eq!(cl_cpu_buf.read(), &[1, 2, 3, 4, 5, 6]);
            assert_eq!(cl_dev.modules.cache.borrow().nodes.len(), 1)
        }

        println!("duration: {dur}");

        Ok(())
    }
}
