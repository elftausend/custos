use custos::{
    cache::CacheReturn, get_count, opencl::construct_buffer, Buffer, GraphReturn, Ident, OpenCL,
    CPU,
};

use super::{AddBuf, AddOp};

#[test]
fn test_access_cached_after_unified_construct_buf() -> custos::Result<()> {
    let cl_dev = OpenCL::new(0)?;

    let a = Buffer::from((&cl_dev, [1, 2, 3, 4, 5]));
    let b = Buffer::from((&cl_dev, [1, 2, 3, 4, 5]));

    // some operation to generate an OpenCL graph
    let c = a.relu();

    assert_eq!(cl_dev.graph().nodes.len(), 3);
    assert_eq!(get_count(), 3);

    let device = CPU::new();
    let no_drop = device.add(&c, &b);

    let cl_cpu_buf = unsafe { construct_buffer(&cl_dev, no_drop, (&c, &b)) }?;

    let cached_cl_cpu_buf = cl_dev
        .cache()
        .nodes
        .get(&Ident {
            idx: cl_cpu_buf.ident.idx,
            len: cl_cpu_buf.len(),
        })
        .unwrap()
        .clone();

    assert_eq!(cl_cpu_buf.ptrs().0 as *mut u8, cached_cl_cpu_buf.host_ptr);
    assert_eq!(cl_cpu_buf.ptrs().1, cached_cl_cpu_buf.ptr);

    Ok(())
}

#[test]
fn test_multiple_construct_buffer() -> custos::Result<()> {
    let cl_dev = OpenCL::new(0)?;

    let device = CPU::new();

    let a = Buffer::from((&cl_dev, [1, 2, 3, 4, 5]));
    let b = Buffer::from((&cl_dev, [1, 2, 3, 4, 5]));
    let c = a.relu();

    let no_drop = device.add(&c, &b);
    let _cl_cpu_buf = unsafe { construct_buffer(&cl_dev, no_drop, (&c, &b)) }?;

    Ok(())
}
