use crate::cache::DynAnyWrapper;
use std::ffi::c_void;

use crate::{AllocFlag, AsAny, Cache, DeviceError, Unit};

use super::CLPtr;
use crate::{
    Base, Buffer, CPU, CachedCPU, CachedModule, Cursor, Device, OpenCL, Shape, UnifiedMemChain,
    UniqueId, WrappedData,
};
use min_cl::api::{MemFlags, create_buffer};

impl<Mods: UnifiedMemChain<Self> + WrappedData> UnifiedMemChain<Self> for OpenCL<Mods> {
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

impl<Mods, CacheType, OclMods, SimpleMods> UnifiedMemChain<OpenCL<OclMods>>
    for CachedModule<Mods, OpenCL<SimpleMods>, CacheType>
where
    CacheType: Cache,
    OclMods: Cursor + WrappedData,
    SimpleMods: WrappedData,
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
            &self.cache,
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

/// Returns an OpenCL pointer that is bound to the host pointer stored in the specified buffer.
/// This function is used in the `constuct_buffer()` function.
/// # Safety
/// The host pointer inside the no_drop `Buffer` must live as long as the resulting pointer.
pub unsafe fn to_cached_unified<OclMods, CpuMods, T, S, CacheType: Cache>(
    device: &OpenCL<OclMods>,
    no_drop: Buffer<T, CPU<CpuMods>, S>,
    cache: &CacheType,
    id: crate::UniqueId,
) -> crate::Result<*mut c_void>
where
    OclMods: WrappedData,
    CpuMods: WrappedData,
    T: Unit + 'static,
    S: Shape,
{
    // use the host pointer to create an OpenCL buffer
    let cl_ptr = unsafe {
        create_buffer(
            device.ctx(),
            MemFlags::MemReadWrite | MemFlags::MemUseHostPtr,
            no_drop.len(),
            Some(&no_drop),
        )?
    };

    // remember: this deallocates the previous pointer!
    cache.insert(
        id,
        no_drop.len(),
        CacheType::CachedValue::new(CLPtr {
            ptr: cl_ptr,
            host_ptr: no_drop.base().ptr,
            len: no_drop.len(),
            flag: AllocFlag::None,
        }),
    );

    Ok(cl_ptr)
}

/// Converts an 'only' CPU buffer into an OpenCL + CPU (unified memory) buffer.
///
/// # Example
#[cfg_attr(unified_cl, doc = "```")]
#[cfg_attr(not(unified_cl), doc = "```ignore")]
/// use custos::prelude::*;
///
/// fn main() -> custos::Result<()> {
///     let cpu = CPU::<Cached<Base>>::new();
///     let device = OpenCL::<Cached<Base>>::new(chosen_cl_idx())?;
///     let mut no_drop: Buffer<f32, _> = cpu.retrieve(4, ()).unwrap();
///     no_drop.write(&[1., 3.1, 2.34, 0.76]);
///
///     let buf = unsafe {
///         construct_buffer(&device, no_drop, &device.modules.cache, 0)?
///     };
///
///     assert_eq!(buf.read(), vec![1., 3.1, 2.34, 0.76]);
///     assert_eq!(buf.read(), &[1., 3.1, 2.34, 0.76]);
///     Ok(())
/// }
/// ```
pub fn construct_buffer<'a, OclMods, CpuMods, T, S>(
    device: &'a OpenCL<OclMods>,
    no_drop: Buffer<'a, T, CPU<CpuMods>, S>,
    cache: &impl Cache,
    id: crate::UniqueId,
) -> crate::Result<Buffer<'a, T, OpenCL<OclMods>, S>>
where
    OclMods: Cursor + WrappedData,
    CpuMods: WrappedData,
    T: Unit + 'static,
    S: Shape,
{
    use crate::PtrType;

    if no_drop.data.flag() == AllocFlag::None {
        return Err(DeviceError::UnifiedConstructInvalidInputBuffer.into());
    }

    unsafe { device.bump_cursor() };

    // if buffer was already converted, return the cache entry.
    if let Ok(rawcl) = cache.get(id, no_drop.len) {
        let rawcl = rawcl
            .as_any()
            .downcast_ref::<<OpenCL<OclMods> as Device>::Base<T, S>>()
            .unwrap();
        let data = device.default_base_to_data::<T, S>(CLPtr {
            ptr: rawcl.ptr,
            host_ptr: rawcl.host_ptr,
            len: no_drop.len(),
            flag: AllocFlag::Wrapper,
        });
        return Ok(Buffer {
            data,
            device: Some(device),
        });
    }
    let (host_ptr, len) = (no_drop.base().ptr, no_drop.len());
    let ptr = unsafe { to_cached_unified(device, no_drop, cache, id)? };

    let data = device.default_base_to_data::<T, S>(CLPtr {
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
        AllocFlag, Base, Buffer, CPU, Cached, Device, DeviceError, FastCache, OpenCL, Retriever,
        UnifiedMemChain,
        opencl::{CLPtr, chosen_cl_idx},
    };

    use super::{construct_buffer, to_cached_unified};

    #[test]
    fn test_unified_mem_chain_ideal() -> crate::Result<()> {
        let cpu = CPU::<Cached<Base>>::new();
        let device = OpenCL::<Cached<Base>>::new(0)?;
        let mut no_drop = cpu.retrieve::<0>(3, ()).unwrap();
        no_drop.write(&[1., 2.3, 0.76]);

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
        let cache = FastCache::new();

        let buf = construct_buffer(&device, no_drop, &cache, 0);
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
        let mut cache = FastCache::new();

        let (host_ptr, len) = (no_drop.data.ptr, no_drop.len());
        let cl_host_ptr = unsafe { to_cached_unified(&device, no_drop, &mut cache, 0)? };

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
        let mut cache = FastCache::new();

        let buf: Buffer<_, _> = construct_buffer(&device, no_drop, &mut cache, 0)?;

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
                &cl_dev.modules.cache,
                cl_dev.cursor() as UniqueId,
            )?;

            dur += start.elapsed().as_secs_f64();

            assert_eq!(cl_cpu_buf.read(), &[1, 2, 3, 4, 5, 6]);
            assert_eq!(cl_cpu_buf.read(), &[1, 2, 3, 4, 5, 6]);
            assert_eq!(cl_dev.modules.cache.nodes.len(), 1)
        }

        println!("duration: {dur}");

        Ok(())
    }

    // improved lifetime annotation rendered this test useless
    /*#[cfg(unified_cl)]
    #[cfg(not(feature = "realloc"))]
    #[test]
    fn test_cpu_to_unified_leak() -> crate::Result<()> {
        use std::{collections::HashMap, hash::BuildHasherDefault, rc::Rc};

        use crate::{range, set_count, Device, Ident, IdentHasher};

        let cl_dev = OpenCL::<Base>::new(chosen_cl_idx())?;

        unsafe { set_count(0) };

        for _ in range(10) {
            let cl_cpu_buf = {
                let cpu = CPU::<Base>::new();
                let mut buf = cpu.retrieve::<i32, ()>(6, ());
                buf.copy_from_slice(&[1, 2, 3, 4, 5, 6]);

                let cl_cpu_buf = unsafe { crate::opencl::construct_buffer(&cl_dev, buf, ())? };
                let mut hm = HashMap::<Ident, _, BuildHasherDefault<IdentHasher>>::default();
                std::mem::swap(&mut cpu.addons.cache.borrow_mut().nodes, &mut hm);
                for mut value in hm {
                    let ptr = Rc::get_mut(&mut value.1).unwrap();
                    ptr.ptr = std::ptr::null_mut();
                }
                cl_cpu_buf
            };
            assert_eq!(cl_cpu_buf.as_slice(), &[1, 2, 3, 4, 5, 6]);
            assert_eq!(cl_cpu_buf.read(), &[1, 2, 3, 4, 5, 6]);
        }
        Ok(())
    }*/
}
