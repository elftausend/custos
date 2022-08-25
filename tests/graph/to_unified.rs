use custos::{CPU, Buffer, CLDevice, opencl::construct_buffer};

#[test]
fn test_to_unified_graph_opt_cl() -> custos::Result<()> {
    let device = CPU::new();

    let buf = Buffer::from((&device, [1, 2, 3, 4, 5]));
    let cl_dev = CLDevice::new(0)?;
    
//    unsafe {construct_buffer(&cl_dev, no_drop, add_node)};

    Ok(())
}