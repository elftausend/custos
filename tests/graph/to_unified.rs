use custos::{CPU, Buffer, CLDevice, opencl::construct_buffer, GraphReturn, cache::CacheReturn, Node, range};

use super::{AddOp, AddBuf};

#[test]
fn test_to_unified_graph_opt_cl() -> custos::Result<()> {
    let cl_dev = CLDevice::new(0)?;

    if !cl_dev.unified_mem() {
        return Ok(());
    }

    let device = CPU::new();

    let _cpu_a = Buffer::from((&device, [1, 2, 3, 4, 5]));
    

    let a = Buffer::from((&cl_dev, [1, 2, 3, 4, 5]));
    let b = Buffer::from((&cl_dev, [1, 2, 3, 4, 5]));

    // some operation to generate an OpenCL graph
    let c = a.relu();

    let no_drop = device.add(&c, &b);

    let cl_cpu_buf = unsafe {construct_buffer(&cl_dev, no_drop, (&c, &b))}?;
    let graph = cl_dev.graph();

    println!("nodes: {:?}", cl_dev.cache.borrow().nodes);

    cl_dev.cache().nodes.get(&Node {
        idx: cl_cpu_buf.node.idx as usize,
        len: cl_cpu_buf.len
    }).unwrap();

    println!("graph: {graph:?}");
    
    //println!("cl_cpu_buf: {:?}", cl_cpu_buf);

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
    let cl_cpu_buf = unsafe {construct_buffer(&cl_dev, no_drop, (&c, &b))}?;
    
    

    Ok(())
}