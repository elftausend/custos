
use custos::{libs::{opencl::{CLDevice, api::OCLError}, cpu::CPU}, Buffer, Device, VecRead};

pub fn get_mut_slice<'a, T>(buf: &mut Buffer<T>) -> &'a mut [T] {
    unsafe {
        std::slice::from_raw_parts_mut(buf.ptr, buf.len)
    }
}

pub fn get_slice<'a, T>(buf: &Buffer<T>) -> &'a [T] {
    unsafe {
        std::slice::from_raw_parts(buf.ptr, buf.len)
    }
}

pub fn read<T, D: Device<T>>(device: D, buf: &Buffer<T>) -> Vec<T> where D: VecRead<T> {
    device.read(buf)
}

#[test]
fn devices() -> Result<(), OCLError> {
    
    let device = CLDevice::get(0)?;
    println!("{}", device.get_name()?);    
    Ok(())
}

#[test]
fn test_buffer_from_read() -> Result<(), OCLError> {
    let buf = Buffer::<f32>::from((&CLDevice::get(0)?, &[3.13, 3., 1., 8.]));
    assert_eq!(read(CLDevice::get(0)?, &buf), vec![3.13, 3., 1., 8.,]);

    let buf = Buffer::<f32>::from((&CPU, &[3.13, 3., 1., 8.]));
    assert_eq!(read(CPU, &buf), vec![3.13, 3., 1., 8.,]);
    Ok(())
}

#[test]
fn test_buffer_alloc_and_read() -> Result<(), OCLError> {
    let mut buf = Buffer::<u8>::new(CPU, 10);
    
    
    let buf_slice = get_mut_slice(&mut buf);
    buf_slice.copy_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    assert_eq!(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], buf_slice);
    
    CPU::drop(buf);
    
    
    let buf = Buffer::<f32>::from((&CLDevice::get(0)?, &[3.13, 3., 1., 8.]));
    let buf_read = read(CLDevice::get(0)?, &buf);
    assert_eq!(&[3.13, 3., 1., 8.], buf_read.as_slice());

    let buf = Buffer::<f32>::from((&CPU, &[3.13, 3., 1., 8.]));
    let buf_read = read(CPU, &buf);
    assert_eq!(&[3.13, 3., 1., 8.], buf_read.as_slice());

    let buf_read = get_slice(&buf);
    assert_eq!(&[3.13, 3., 1., 8.], buf_read);

    Ok(())   
}
