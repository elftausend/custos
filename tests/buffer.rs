use std::ffi::c_void;

use custos::cpu::cpu_cached;
#[cfg(feature = "opencl")]
use custos::{libs::opencl::CLDevice, Error};
use custos::{Buffer, Alloc, VecRead};

use custos::CPU;

#[cfg(not(feature="realloc"))]
use custos::{get_count, set_count};

pub fn get_mut_slice<'a, T>(buf: &'a mut Buffer<T>) -> &'a mut [T] {
    unsafe { std::slice::from_raw_parts_mut(buf.ptr.0, buf.len) }
}

pub fn get_slice<'a, T>(buf: &'a Buffer<T>) -> &'a [T] {
    unsafe { std::slice::from_raw_parts(buf.ptr.0, buf.len) }
}

pub fn read<T, D: Alloc<T>>(device: &D, buf: &Buffer<T>) -> Vec<T>
where
    D: VecRead<T>,
{
    device.read(&buf)
}

#[cfg(feature = "opencl")]
#[test]
fn test_cldevice_name() -> Result<(), Error> {
    let device = CLDevice::new(0)?;
    println!("{}", device.name()?);

    Ok(())
}

#[cfg(feature = "opencl")]
#[test]
fn test_cldevice_version() -> Result<(), Error> {
    let device = CLDevice::new(0)?;
    println!("{}", device.version()?);
    Ok(())
}

#[cfg(feature = "opencl")]
#[test]
fn test_cldevice_mem() -> Result<(), Error> {
    let device = CLDevice::new(0)?;
    println!(
        "get_global_mem_size_in_gb: {}",
        device.global_mem_size_in_gb()?
    );
    println!("get_max_mem_alloc_in_gb: {}", device.max_mem_alloc_in_gb()?);
    Ok(())
}

#[cfg(feature = "opencl")]
#[test]
fn test_buffer_from_read() -> Result<(), Error> {
    let device = CLDevice::new(0)?;

    let buf = Buffer::<f32>::from((&device, [3.13, 3., 1., 8.]));
    assert_eq!(read(&device, &buf), vec![3.13, 3., 1., 8.,]);

    let device = CPU::new();

    let buf = Buffer::<f32>::from((&device, [3.13, 3., 1., 8.]));
    assert_eq!(read(&device, &buf), vec![3.13, 3., 1., 8.,]);
    Ok(())
}

#[cfg(feature = "opencl")]
#[test]
fn test_buffer_alloc_and_read() -> Result<(), Error> {
    let device = CPU::new();

    let mut buf = Buffer::<u8>::new(&device, 10);

    let buf_slice = get_mut_slice(&mut buf);
    buf_slice.copy_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    assert_eq!(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], buf_slice);

    let cl = CLDevice::new(0)?;

    let buf = Buffer::<f32>::from((&cl, [3.13, 3., 1., 8.]));
    let buf_read = read(&cl, &buf);
    assert_eq!(&[3.13, 3., 1., 8.], buf_read.as_slice());

    let buf = Buffer::<f32>::from((&device, [3.13, 3., 1., 8.]));
    let buf_read = read(&device, &buf);
    assert_eq!(&[3.13, 3., 1., 8.], buf_read.as_slice());

    let buf_read = get_slice(&buf);
    assert_eq!(&[3.13, 3., 1., 8.], buf_read);

    Ok(())
}

#[test]
fn test_use_number() {
    let num = Box::into_raw(Box::new(10));

    let num = unsafe { Box::from_raw(num) };
    assert_eq!(num, Box::new(10));
}

#[cfg(not(feature = "realloc"))]
#[test]
fn test_cached_cpu() {
    std::env::set_var("RUST_BACKTRACE", "1");
    let device = CPU::new();

    assert_eq!(0, get_count());

    let mut buf = cpu_cached::<f32>(&device, 10);

    assert_eq!(1, get_count());

    for value in buf.as_mut_slice() {
        *value = 1.5;
    }

    let new_buf = cpu_cached::<i32>(&device, 10);
    assert_eq!(device.read(&new_buf), vec![0; 10]);
    assert_eq!(2, get_count());

    set_count(0);
    assert_eq!(0, get_count());
    let buf = cpu_cached::<f32>(&device, 10);

    assert_eq!(device.read(&buf), vec![1.5; 10]);
}

#[cfg(not(feature = "realloc"))]
#[cfg(not(target_os = "linux"))]
#[cfg(feature = "opencl")]
#[test]
fn test_cached_cl() -> Result<(), custos::Error> {
    use custos::opencl::{api::{enqueue_write_buffer, wait_for_event}, cl_cached};

    let device = CLDevice::new(0)?;
    let _k = Buffer::<f32>::new(&device, 1);

    assert_eq!(0, get_count());

    let buf = cl_cached::<f32>(&device, 10);

    assert_eq!(1, get_count());

    unsafe {
        let event = enqueue_write_buffer(&device.queue(), buf.ptr.1, &[0.1f32; 10], true)?;
        wait_for_event(event)?
    }
    assert_eq!(device.read(&buf), vec![0.1; 10]);

    let new_buf = cl_cached::<i32>(&device, 10);
    
    assert_eq!(device.read(&new_buf), vec![0; 10]);
    assert_eq!(2, get_count());

    set_count(0);
    assert_eq!(0, get_count());
    let buf = cl_cached::<f32>(&device, 10);
    println!("new_buf: {buf:?}");
    assert_eq!(device.read(&buf), vec![0.1; 10]);
    Ok(())
}

#[test]
fn test_from_ptrs() {
    let mut value = 4f32;
    let ptr: *mut c_void = &mut value as *mut f32 as *mut c_void;

    let buf = Buffer::<f32>::from((ptr, 5));
    assert_eq!(buf.ptr.0, std::ptr::null_mut());

    let ptr: *mut f32 = &mut value as *mut f32;
    let buf = Buffer::<f32>::from((ptr, 5));
    assert_eq!(buf.ptr.1, std::ptr::null_mut());
}

#[test]
fn test_size_buf() {
    let x = core::mem::size_of::<Buffer<i8>>();
    println!("x: {x}");
}

fn _slice_add<T: Copy + std::ops::Add<Output = T>>(a: &[T], b: &[T], c: &mut [T]) {
    for i in 0..c.len() {
        c[i] = a[i] + b[i]
    }
}

#[test]
fn test_iterate() {
    let cmp = [1f32, 2., 3.3];

    let device = CPU::new();
    let x = Buffer::from((&device, [1., 2., 3.3]));

    for (x, y) in x.iter().zip(cmp) {
        assert!(*x == y)
    }
}

#[cfg(feature = "opencl")]
#[test]
fn test_debug_print_buf() -> custos::Result<()> {
    let device = CLDevice::new(0)?;

    let a = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));

    println!("a: {a:?}");
    Ok(())
}

#[test]
fn test_slice() {
    let device = CPU::new();

    let buf = Buffer::from((&device, [1, 2, 3, 4, 5, 6,]));
    println!("buf: {:?}", buf.as_slice());
}

#[test]
fn test_alloc() {
    let device = CPU::new();
    let buf = cpu_cached::<f32>(&device, 100);
    assert_eq!(buf.read(), vec![0.; 100]);

    let buf = cpu_cached::<f32>(&device, 100);
    assert_eq!(buf.read(), vec![0.; 100]);
    drop(buf);
}

#[test]
fn test_deviceless_buf() {
    let mut buf = {
        let device = CPU::new();
        Buffer::<u8>::deviceless(&device, 5)
    };

    for (idx, element) in buf.iter_mut().enumerate() {
        *element = idx as u8;
    }

    assert_eq!(buf.as_slice(), &[0, 1, 2, 3, 4]);
}

#[test]
#[should_panic]
fn test_deviceless_buf_panic() {
    let buf = {
        let device = CPU::new();
        Buffer::<u8>::deviceless(&device, 5)
    };
    buf.read();
}