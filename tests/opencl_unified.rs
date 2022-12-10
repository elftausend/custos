#![allow(unused)]
use custos::{
    cache::Cache,
    opencl::api::{clCreateBuffer, enqueue_map_buffer, CommandQueue, MemFlags, OCLErrorKind},
    range, set_count, Buffer, Error, OpenCL, Read, CPU,
};
use std::{collections::HashMap, ffi::c_void};

pub fn unified_mem<T>(device: &OpenCL, arr: &mut [T]) -> Result<*mut c_void, Error> {
    let mut err = 0;

    let r = unsafe {
        clCreateBuffer(
            device.ctx().0,
            MemFlags::MemReadWrite | MemFlags::MemCopyHostPtr,
            arr.len() * core::mem::size_of::<T>(),
            arr.as_mut_ptr() as *mut c_void,
            &mut err,
        )
    };

    device.inner.borrow_mut().ptrs.push(r);

    if err != 0 {
        return Err(Error::from(OCLErrorKind::from_value(err)));
    }
    Ok(r)
}

pub fn unified_ptr<T>(cq: &CommandQueue, ptr: *mut c_void, len: usize) -> Result<*mut T, Error> {
    unsafe { enqueue_map_buffer::<T>(&cq, ptr, true, 2 | 1, 0, len).map(|ptr| ptr as *mut T) }
}

#[cfg(feature = "opencl")]
#[test]
fn test_unified_mem_bool() -> Result<(), Error> {
    let device = OpenCL::new(0)?;
    let um = device.unified_mem();
    println!("um: {um}");
    Ok(())
}

#[cfg(feature = "opencl")]
#[test]
fn test_unified_mem() -> Result<(), Error> {
    const TIMES: usize = 10000;
    use std::time::Instant;

    use custos::opencl::api::{create_buffer, release_mem_object, MemFlags};

    let len = 20000;

    let data = vec![1f32; len];

    let device = OpenCL::new(0)?;

    if device.unified_mem() {
        let before = Instant::now();
        for _ in 0..TIMES {
            //std::thread::sleep(std::time::Duration::from_secs(1));

            let buf = create_buffer(
                &device.ctx(),
                MemFlags::MemReadWrite | MemFlags::MemUseHostPtr,
                len,
                Some(&data),
            )?;

            let ptr = unified_ptr::<f32>(&device.queue(), buf, len)?;

            let slice = unsafe { std::slice::from_raw_parts_mut(ptr, len) };

            for idx in 20..100 {
                slice[idx] = 4.;
            }

            unsafe {
                release_mem_object(buf)?;
            }

            // 'data' vec is not freed
            assert_eq!(slice[25], 4.);

            /*
            let mut read = vec![0f32; len];
            let event = unsafe { custos::opencl::api::enqueue_read_buffer(&device.queue(), buf, &mut read, true)}?;
            custos::opencl::api::wait_for_event(event)?;
            println!("read: {read:?}");
            */
        }

        let after = Instant::now();
        println!("use host ptr: {:?}", (after - before) / TIMES as u32);

        let before = Instant::now();
        for _ in 0..TIMES {
            let buf = create_buffer(
                &device.ctx(),
                MemFlags::MemReadWrite | MemFlags::MemCopyHostPtr,
                len,
                Some(&data),
            )?;
            let ptr = unified_ptr::<f32>(&device.queue(), buf, len)?;
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr, len) };

            for idx in 20..100 {
                slice[idx] = 4.;
            }

            unsafe {
                release_mem_object(buf)?;
            }
        }
        let after = Instant::now();
        println!("copy host ptr: {:?}", (after - before) / TIMES as u32);
    }

    /*
    let before = Instant::now();
    for _ in 0..TIMES {
        let buf = create_buffer::<f32>(&device.ctx(), MemFlags::MemReadWrite as u64, len, None)?;

        unsafe {
            release_mem_object(buf)?;
        }
    }

    let after = Instant::now();
    println!("alloc: {:?}", (after-before) / TIMES as u32);
    */

    Ok(())
}

#[cfg(feature = "opencl")]
fn slice_add<T: Copy + std::ops::Add<Output = T>>(a: &[T], b: &[T], c: &mut [T]) {
    for i in 0..c.len() {
        c[i] = a[i] + b[i]
    }
}

/*
#[cfg(feature = "opencl")]
#[test]
fn test_unified_mem_ops() -> Result<(), custos::Error> {
    let device = CLDevice::new(0)?;

    if !device.unified_mem() {
        return Ok(());
    }

    let a = Buffer::from((&device, [1, 4, 3, 2, 7, 9]));
    assert!(a.ptr.0 != std::ptr::null_mut());

    let b = Buffer::from((&device, [2, 1, 7, 4, 3, 2]));
    let mut c = Buffer::from((&device, [0; 6]));

    slice_add(&a, &b, &mut c);

    let res = device.read(&c);
    assert_eq!(res, c.as_slice().to_vec());

    Ok(())
}

#[cfg(feature = "opencl")]
#[test]
fn test_unified_mem_iterate() -> custos::Result<()> {
    let device = CLDevice::new(0)?;

    if !device.unified_mem() {
        println!("CLDevice uses own memory");
        return Ok(());
    }

    let mut a = Buffer::from((&device, [1, 2, 3, 4, 5]));

    for value in &mut a {
        *value += 2;
    }

    let cl_data = device.read(&a);
    assert_eq!(a.as_slice(), &cl_data);
    assert_eq!(&cl_data, &[3, 4, 5, 6, 7,]);

    Ok(())
}*/

#[cfg(unified_cl)]
#[cfg(not(feature = "realloc"))]
#[test]
fn test_cpu_to_unified() -> custos::Result<()> {
    let device = CPU::new();
    let mut buf = Cache::get::<i32, 0>(&device, 6, ());
    buf.copy_from_slice(&[1, 2, 3, 4, 5, 6]);

    let cl_dev = OpenCL::new(0)?;
    let cl_cpu_buf = unsafe { custos::opencl::construct_buffer(&cl_dev, buf, ())? };

    assert_eq!(cl_cpu_buf.as_slice(), &[1, 2, 3, 4, 5, 6]);
    assert_eq!(cl_cpu_buf.read(), &[1, 2, 3, 4, 5, 6]);

    Ok(())
}

#[cfg(unified_cl)]
#[cfg(not(feature = "realloc"))]
#[test]
fn test_cpu_to_unified_leak() -> custos::Result<()> {
    use std::rc::Rc;

    let cl_dev = OpenCL::new(0)?;

    set_count(0);

    for _ in range(10) {
        let cl_cpu_buf = {
            let cpu = CPU::new();
            let mut buf = Cache::get::<i32, 0>(&cpu, 6, ());
            buf.copy_from_slice(&[1, 2, 3, 4, 5, 6]);

            let cl_cpu_buf = unsafe { custos::opencl::construct_buffer(&cl_dev, buf, ())? };
            let mut hm = HashMap::new();
            std::mem::swap(&mut cpu.cache.borrow_mut().nodes, &mut hm);
            for mut value in hm {
                let mut ptr = Rc::get_mut(&mut value.1).unwrap();
                ptr.ptr = std::ptr::null_mut();
            }
            cl_cpu_buf
        };
        assert_eq!(cl_cpu_buf.as_slice(), &[1, 2, 3, 4, 5, 6]);
        assert_eq!(cl_cpu_buf.read(), &[1, 2, 3, 4, 5, 6]);
    }
    Ok(())
}

#[cfg(unified_cl)]
#[cfg(not(feature = "realloc"))]
#[test]
fn test_cpu_to_unified_perf() -> custos::Result<()> {
    use std::time::Instant;

    let cl_dev = OpenCL::new(0)?;
    let device = CPU::new();

    let mut dur = 0.;

    for _ in range(100) {
        let mut buf = Cache::get::<i32, 0>(&device, 6, ());

        buf.copy_from_slice(&[1, 2, 3, 4, 5, 6]);

        let start = Instant::now();
        let cl_cpu_buf = unsafe { custos::opencl::construct_buffer(&cl_dev, buf, ())? };
        dur += start.elapsed().as_secs_f64();

        assert_eq!(cl_cpu_buf.as_slice(), &[1, 2, 3, 4, 5, 6]);
        assert_eq!(cl_cpu_buf.read(), &[1, 2, 3, 4, 5, 6]);
    }

    println!("duration: {dur}");

    Ok(())
}
