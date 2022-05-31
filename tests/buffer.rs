

#[cfg(not(feature="safe"))]
use custos::{AsDev, CPU};

use custos::{Buffer, Device, VecRead};
#[cfg(feature="opencl")]
use custos::{libs::opencl::CLDevice, Error};

#[cfg(not(feature="safe"))]
use custos::{cached, get_count, set_count};

pub fn get_mut_slice<T>(buf: &mut Buffer<T>) -> &mut [T] {
    unsafe {
        std::slice::from_raw_parts_mut(buf.ptr.0, buf.len)
    }
}

pub fn get_slice<T>(buf: &Buffer<T>) -> &[T] {
    unsafe {
        std::slice::from_raw_parts(buf.ptr.0, buf.len)
    }
}

pub fn read<T, D: Device<T>>(device: &D, buf: &Buffer<T>) -> Vec<T> where D: VecRead<T> {
    device.read(&buf)
}

#[cfg(feature="opencl")]
#[test]
fn test_cldevice_name() -> Result<(), Error> {
    let device = CLDevice::get(0)?;
    println!("{}", device.name()?);

    Ok(())
}

#[cfg(feature="opencl")]
#[test]
fn test_cldevice_version() -> Result<(), Error> {
    let device = CLDevice::get(0)?;
    println!("{}", device.version()?);
    Ok(())
}

#[cfg(feature="opencl")]
#[test]
fn test_cldevice_mem() -> Result<(), Error> {
    let device = CLDevice::get(0)?;
    println!("get_global_mem_size_in_gb: {}", device.global_mem_size_in_gb()?);
    println!("get_max_mem_alloc_in_gb: {}", device.max_mem_alloc_in_gb()?);
    Ok(())
}

#[cfg(feature="opencl")]
#[cfg(feature="safe")]
use custos::CPU;

#[cfg(feature="opencl")]
#[test]
fn test_buffer_from_read() -> Result<(), Error> {
    use custos::AsDev;

    let device = CLDevice::get(0)?.select();

    let buf = Buffer::<f32>::from((&device, [3.13, 3., 1., 8.]));
    assert_eq!(read(&device, &buf), vec![3.13, 3., 1., 8.,]);

    let device = CPU::new().select();

    let buf = Buffer::<f32>::from((&device, [3.13, 3., 1., 8.]));
    assert_eq!(read(&device, &buf), vec![3.13, 3., 1., 8.,]);
    Ok(())
}

#[cfg(feature="opencl")]
#[test]
fn test_buffer_alloc_and_read() -> Result<(), Error> {
    use custos::AsDev;

    let device = CPU::new().select();

    let mut buf = Buffer::<u8>::new(&device, 10);
    
    let buf_slice = get_mut_slice(&mut buf);
    buf_slice.copy_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    assert_eq!(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], buf_slice);
    
    let cl = CLDevice::get(0)?.select();

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

    let num = unsafe {
        Box::from_raw(num)
    };
    assert_eq!(num, Box::new(10));
}

#[cfg(not(feature="safe"))]
#[test]
fn test_cached_cpu() {
    let device = CPU::new().select();
    
    assert_eq!(0, get_count());
    
    let mut buf = cached::<f32>(10);
    
    assert_eq!(1, get_count());
    
    for value in buf.as_mut_slice() {
        *value = 1.5;
    }

    let new_buf = cached::<i32>(10);
    assert_eq!(device.read(&new_buf), vec![0; 10]);
    assert_eq!(2, get_count());

    set_count(0);
    let buf = cached::<f32>(10);
    
    assert_eq!(device.read(&buf), vec![1.5; 10]);
}


#[cfg(not(feature="safe"))]
#[cfg(feature="opencl")]
#[test]
fn test_cached_cl() -> Result<(), custos::Error> {
    use custos::opencl::api::{enqueue_write_buffer, wait_for_event};

    let device = CLDevice::get(0)?.select();
    let _ = Buffer::<f32>::new(&device, 1);
    
    assert_eq!(0, get_count());
    
    let buf = cached::<f32>(10);
    
    assert_eq!(1, get_count());
    
    unsafe {
        let event = enqueue_write_buffer(&device.queue(), buf.ptr.1, &[0.1f32; 10], true)?;
        wait_for_event(event)?
    }
    
    let new_buf = cached::<i32>(10);
    assert_eq!(device.read(&new_buf), vec![0; 10]);
    assert_eq!(2, get_count());
    
    set_count(0);
    let buf = cached::<f32>(10);
    
    assert_eq!(device.read(&buf), vec![0.1; 10]);
    Ok(())
}


#[test]
fn test_size_buf() {
    let _device = CPU::new();

    let x = core::mem::size_of::<Buffer<i8>>();
    println!("x: {x}");
}