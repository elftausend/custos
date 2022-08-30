use custos::{CPU, Buffer, CLDevice, opencl::construct_buffer, cache::CacheReturn, Ident};

use super::{AddOp, AddBuf};

#[test]
fn test_access_cached_after_unified_construct_buf() -> custos::Result<()> {
    let cl_dev = CLDevice::new(0)?;

    if !cl_dev.unified_mem() {
        return Ok(());
    }
    
    let a = Buffer::from((&cl_dev, [1, 2, 3, 4, 5]));
    let b = Buffer::from((&cl_dev, [1, 2, 3, 4, 5]));

    // some operation to generate an OpenCL graph
    let c = a.relu();

    let device = CPU::new();
    let no_drop = device.add(&c, &b);

    let cl_cpu_buf = unsafe {construct_buffer(&cl_dev, no_drop, (&c, &b))}?;

    println!("nodes: {:?}", cl_dev.cache.borrow().nodes);

    let cached_cl_cpu_buf = cl_dev.cache().nodes.get(&Ident {
        idx: cl_cpu_buf.node.ident_idx as usize,
        len: cl_cpu_buf.len
    }).unwrap().clone();

    assert_eq!(cl_cpu_buf.ptr.0 as *mut u8, cached_cl_cpu_buf.host_ptr);
    assert_eq!(cl_cpu_buf.ptr.1, cached_cl_cpu_buf.ptr);


    Ok(())
}

#[test]
fn test_multiple_construct_buffer()  -> custos::Result<()> {
    let cl_dev = CLDevice::new(0)?;

    if !cl_dev.unified_mem() {
        return Ok(());
    }
    let device = CPU::new();


    let a = Buffer::from((&cl_dev, [1, 2, 3, 4, 5]));
    let b = Buffer::from((&cl_dev, [1, 2, 3, 4, 5]));
    let c = a.relu();

    let no_drop = device.add(&c, &b);
    let _cl_cpu_buf = unsafe {construct_buffer(&cl_dev, no_drop, (&c, &b))}?;
    
    

    Ok(())
}