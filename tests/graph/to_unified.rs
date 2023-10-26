/* 
use custos::{
    opencl::construct_buffer, Buffer, OpenCL,
    CPU, prelude::chosen_cl_idx, Base, Graph, Cached,
};

use super::{AddBuf, AddOp};

#[test]
fn test_access_cached_after_unified_construct_buf() -> custos::Result<()> {
    let cl_dev = OpenCL::<Graph<Cached<Base>>>::new(chosen_cl_idx())?;

    let a = Buffer::from((&cl_dev, [1, 2, 3, 4, 5]));
    let b = Buffer::from((&cl_dev, [1, 2, 3, 4, 5]));

    // some operation to generate an OpenCL graph
    let c = a.relu();

    assert_eq!(cl_dev.modules.graph_trans.borrow().opt_graph.nodes.len(), 3);

    let device = CPU::<Base>::new();
    let no_drop = device.add(&c, &b);
    
    let cache = cl_dev
        .modules
        .modules
        .cache
        .borrow_mut();

    let cl_cpu_buf = unsafe { construct_buffer(&cl_dev, no_drop, cache,(&c, &b)) }?;

    let cached_cl_cpu_buf = cache
        .nodes
        .get(&Ident {
            idx: cl_cpu_buf.ident.unwrap().idx,
            len: cl_cpu_buf.len(),
        })
        .unwrap()
        .clone();

    assert_eq!(cl_cpu_buf.ptrs().0 as *mut u8, cached_cl_cpu_buf.host_ptr);
    assert_eq!(cl_cpu_buf.ptrs().1, cached_cl_cpu_buf.ptr);

    Ok(())
}
*/
