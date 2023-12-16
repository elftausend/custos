use core::{hash::BuildHasherDefault, panic::Location};
use std::{collections::HashMap, ffi::c_void, rc::Rc};

#[cfg(not(feature = "realloc"))]
use crate::{AllocFlag, DeviceError};

use super::CLPtr;
use crate::{
    Base, Buffer, CachedCPU, CachedModule, Device, HashLocation, LocationHasher, OnDropBuffer,
    OpenCL, Shape, UnifiedMemChain, CPU,
};
use min_cl::api::{create_buffer, MemFlags};

impl<Mods: UnifiedMemChain<Self> + OnDropBuffer> UnifiedMemChain<Self> for OpenCL<Mods> {
    #[inline]
    fn construct_unified_buf_from_cpu_buf<'a, T: 'static, S: Shape>(
        &self,
        device: &'a Self,
        no_drop_buf: Buffer<'a, T, CachedCPU, S>,
    ) -> crate::Result<Buffer<'a, T, Self, S>> {
        self.modules
            .construct_unified_buf_from_cpu_buf(device, no_drop_buf)
    }
}

impl<Mods, OclMods: OnDropBuffer, SimpleMods: OnDropBuffer> UnifiedMemChain<OpenCL<OclMods>>
    for CachedModule<Mods, OpenCL<SimpleMods>>
{
    #[inline]
    fn construct_unified_buf_from_cpu_buf<'a, T: 'static, S: Shape>(
        &self,
        device: &'a OpenCL<OclMods>,
        no_drop_buf: Buffer<'a, T, CachedCPU, S>,
    ) -> crate::Result<Buffer<'a, T, OpenCL<OclMods>, S>> {
        construct_buffer(
            device,
            no_drop_buf,
            &mut self.cache.borrow_mut().nodes,
            Location::caller().into(),
        )
    }
}

impl<D: Device> UnifiedMemChain<D> for Base {
    #[inline]
    fn construct_unified_buf_from_cpu_buf<'a, T, S: Shape>(
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
pub unsafe fn to_cached_unified<OclMods: OnDropBuffer, CpuMods: OnDropBuffer, T, S: Shape>(
    device: &OpenCL<OclMods>,
    no_drop: Buffer<T, CPU<CpuMods>, S>,
    cache: &mut HashMap<
        HashLocation<'static>,
        Rc<dyn core::any::Any>,
        BuildHasherDefault<LocationHasher>,
    >,
    location: HashLocation<'static>,
) -> crate::Result<*mut c_void> {
    // use the host pointer to create an OpenCL buffer
    let cl_ptr = create_buffer(
        device.ctx(),
        MemFlags::MemReadWrite | MemFlags::MemUseHostPtr,
        no_drop.len(),
        Some(&no_drop),
    )?;

    let old_ptr = cache.insert(
        location,
        Rc::new(CLPtr {
            ptr: cl_ptr,
            host_ptr: no_drop.base().ptr as *mut u8,
            len: no_drop.len(),
            flag: AllocFlag::None,
        }),
    );

    // this pointer was overwritten previously, hence can it be deallocated
    // this line can be removed, however it shows that deallocating the old pointer makes sense
    drop(old_ptr);

    Ok(cl_ptr)
}

#[cfg(not(feature = "realloc"))]
/// Converts an 'only' CPU buffer into an OpenCL + CPU (unified memory) buffer.
///
/// # Example
#[cfg_attr(unified_cl, doc = "```")]
#[cfg_attr(not(unified_cl), doc = "```ignore")]
/// use custos::prelude::*;
///
/// fn main() -> custos::Result<()> {
///     let cpu = CPU::<Cached<Base>>::new();
///     let mut no_drop: Buffer<f32, _> = cpu.retrieve(4, ());
///     no_drop.write(&[1., 3.1, 2.34, 0.76]);
///     
///     let device = OpenCL::<Cached<Base>>::new(chosen_cl_idx())?;
///     let buf = unsafe {
///         construct_buffer(&device, no_drop, &mut device.modules.cache.borrow_mut().nodes, std::panic::Location::caller().into())?
///     };
///     
///     assert_eq!(buf.read(), vec![1., 3.1, 2.34, 0.76]);
///     assert_eq!(buf.as_slice(), &[1., 3.1, 2.34, 0.76]);
///     Ok(())
/// }
/// ```
pub fn construct_buffer<'a, OclMods: OnDropBuffer, CpuMods: OnDropBuffer, T: 'static, S: Shape>(
    device: &'a OpenCL<OclMods>,
    no_drop: Buffer<'a, T, CPU<CpuMods>, S>,
    cache: &mut HashMap<
        HashLocation<'static>,
        Rc<dyn core::any::Any>,
        BuildHasherDefault<LocationHasher>,
    >,
    location: HashLocation<'static>,
) -> crate::Result<Buffer<'a, T, OpenCL<OclMods>, S>> {
    use crate::PtrType;

    if no_drop.data.flag() == AllocFlag::None {
        return Err(DeviceError::UnifiedConstructInvalidInputBuffer.into());
    }

    // if buffer was already converted, return the cache entry.
    if let Some(rawcl) = cache.get(&location) {
        let rawcl = rawcl.downcast_ref::<<OpenCL::<OclMods> as Device>::Base<T, S>>().unwrap();
        let data = device.base_to_data::<T, S>(CLPtr {
            ptr: rawcl.ptr,
            host_ptr: rawcl.host_ptr as *mut T,
            len: no_drop.len(),
            flag: no_drop.data.flag(),
        });
        return Ok(Buffer {
            data,
            device: Some(device),
        });
    }
    let (host_ptr, len) = (no_drop.base().ptr, no_drop.len());
    let ptr = unsafe { to_cached_unified(device, no_drop, cache, location)? };

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
        AllocFlag, Base, Buffer, Cache, Cached, Device, DeviceError, HashLocation, HostPtr, OpenCL,
        Retriever, UnifiedMemChain, CPU,
    };

    use super::{construct_buffer, to_cached_unified};

    #[test]
    fn test_unified_mem_chain_ideal() -> crate::Result<()> {
        let cpu = CPU::<Cached<Base>>::new();
        let mut no_drop = cpu.retrieve::<0>(3, ());
        no_drop.write(&[1., 2.3, 0.76]);

        let device = OpenCL::<Cached<Base>>::new(0)?;
        let buf = device.construct_unified_buf_from_cpu_buf::<_, ()>(&device, no_drop)?;
        assert_eq!(buf.as_slice(), &[1., 2.3, 0.76]);
        Ok(())
    }

    #[test]
    fn test_unified_mem_chain_ideal_using_cpu_from_opencl_dev() -> crate::Result<()> {
        let device = OpenCL::<Cached<Base>>::new(0)?;

        let mut no_drop = device.cpu.retrieve::<0>(3, ());
        no_drop.write(&[1., 2.3, 0.76]);

        let buf = device.construct_unified_buf_from_cpu_buf::<_, ()>(&device, no_drop)?;

        assert_eq!(buf.as_slice(), &[1., 2.3, 0.76]);
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
        let mut no_drop = cpu.retrieve::<0>(3, ());
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
        let mut no_drop = cpu.retrieve::<0>(3, ());
        no_drop.write(&[1., 2.3, 0.76]);

        let device = OpenCL::<Base>::new(chosen_cl_idx())?;
        let mut cache = Cache::new();

        let buf = construct_buffer(&device, no_drop, &mut cache.nodes, HashLocation::here());
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
        let mut no_drop: Buffer<_, _> = cpu.retrieve::<0>(3, ());
        no_drop.write(&[1., 2.3, 0.76]);

        let device = OpenCL::<Base>::new(chosen_cl_idx())?;
        let mut cache = Cache::new();

        let (host_ptr, len) = (no_drop.data.ptr, no_drop.len());
        let cl_host_ptr =
            unsafe { to_cached_unified(&device, no_drop, &mut cache.nodes, HashLocation::here())? };

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
        assert_eq!(buf.as_slice(), &[1., 2.3, 0.76]);
        Ok(())
    }

    #[test]
    fn test_construct_buffer() -> crate::Result<()> {
        let cpu = CPU::<Cached<Base>>::new();
        let mut no_drop = cpu.retrieve::<0>(3, ());
        no_drop.write(&[1., 2.3, 0.76]);

        let device = OpenCL::<Base>::new(chosen_cl_idx())?;
        let mut cache = Cache::new();

        let buf: Buffer<_, _> =
            construct_buffer(&device, no_drop, &mut cache.nodes, HashLocation::here())?;

        assert_eq!(buf.read(), vec![1., 2.3, 0.76]);
        assert_eq!(buf.as_slice(), &[1., 2.3, 0.76]);

        Ok(())
    }

    #[cfg(unified_cl)]
    #[test]
    fn test_cpu_to_unified_is_reusing_converted_buf() -> crate::Result<()> {
        use std::time::Instant;

        use crate::{Base, Cached, HashLocation, Retriever};

        let cl_dev = OpenCL::<Cached<Base>>::new(0)?;
        let device = CPU::<Cached<Base>>::new();

        let mut dur = 0.;

        for _ in 0..100 {
            let mut buf: Buffer<i32,  _> = device.retrieve::<0>(6, ());

            buf.copy_from_slice(&[1, 2, 3, 4, 5, 6]);

            let start = Instant::now();
            let cl_cpu_buf = construct_buffer(
                &cl_dev,
                buf,
                &mut cl_dev.modules.cache.borrow_mut().nodes,
                HashLocation::here(),
            )?;
            dur += start.elapsed().as_secs_f64();

            assert_eq!(cl_cpu_buf.as_slice(), &[1, 2, 3, 4, 5, 6]);
            assert_eq!(cl_cpu_buf.read(), &[1, 2, 3, 4, 5, 6]);
            assert_eq!(cl_dev.modules.cache.borrow().nodes.len(), 1)
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
