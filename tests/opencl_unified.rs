#[cfg(feature="opencl")]
use custos::{CLDevice, Error};

#[cfg(feature="opencl")]
#[test]
fn test_unified_mem_bool() -> Result<(), Error> {
    let device = CLDevice::get(0)?;
    let um = device.unified_mem()?;
    println!("um: {um}");
    Ok(())
}

#[cfg(feature="opencl")]
#[test]
fn test_unified_mem() -> Result<(), Error> {
    use custos::opencl::api::{create_buffer, MemFlags, enqueue_map_buffer};

    let data = vec![1f32; 20];

    let device = CLDevice::get(0)?;
    let buf = create_buffer(&device.ctx(), MemFlags::MemReadWrite | MemFlags::MemUseHostPtr, 20, Some(&data))?;
    let ptr = unsafe { enqueue_map_buffer::<f32>(&device.queue(), buf, true, 2, 0, 20)}? as *mut f32;
    let slice = unsafe {std::slice::from_raw_parts_mut(ptr, 20)};
    slice[0] = 4.;

    //unsafe { enqueue_read_buffer(cq, mem, data, block)};

    Ok(())
}


