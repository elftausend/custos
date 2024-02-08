use custos::{Base, Buffer, Cached, Cursor, Graph, OptimizeMemGraph, CPU};

#[cfg(feature = "opencl")]
use custos::OpenCL;

use crate::graph::AddOp;

#[test]
fn test_graph() -> custos::Result<()> {
    let device = CPU::<Graph<Cached<Base>>>::new();

    // idx: 0
    let a = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));
    // idx: 1
    let b = Buffer::from((&device, [2, 3, 1, 4, 0, 5]));

    for ep in device.range(0..=1) {
        // idx: 2, deps: [0, 1]
        let c = a.add(&b);
        assert_eq!(vec![3, 5, 4, 8, 5, 11], c.read());

        // idx: 3, deps: [2, 2]
        let d = c.relu();

        assert_eq!(vec![3, 5, 4, 8, 5, 11], d.read());

        // idx: 4, deps: [3, 1]
        let e = d.add(&b);

        if ep == 1 {
            assert_eq!(c.ptr, d.ptr);
            assert_eq!(c.ptr, e.ptr);
        }
        device.optimize_mem_graph(&device, None).unwrap();
    }
    Ok(())
}

#[cfg(feature = "opencl")]
#[test]
fn test_graph_cl() -> custos::Result<()> {
    use custos::prelude::chosen_cl_idx;

    let device = OpenCL::<Graph<Cached<Base>>>::new(chosen_cl_idx())?;

    // idx: 0
    let a = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));
    // idx: 1
    let b = Buffer::from((&device, [2, 3, 1, 4, 0, 5]));

    for ep in device.range(0..=1) {
        // idx: 2, deps: [0, 1]
        let c = a.add(&b);
        assert_eq!(vec![3, 5, 4, 8, 5, 11], c.read());

        // idx: 3, deps: [2, 2]
        let d = c.relu();

        assert_eq!(vec![3, 5, 4, 8, 5, 11], d.read());

        // idx: 4, deps: [3, 1]
        let e = d.add(&b);

        if ep == 1 {
            assert_eq!(c.ptr, d.ptr);
            assert_eq!(c.ptr, e.ptr);
        }
        device.optimize_mem_graph(&device, None).unwrap();
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_graph_cu() -> custos::Result<()> {
    use custos::{Cached, Cursor, CUDA};

    let device = CUDA::<Graph<Cached<Base>>>::new(0)?;

    // idx: 0
    let a = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));
    // idx: 1
    let b = Buffer::from((&device, [2, 3, 1, 4, 0, 5]));

    for ep in device.range(0..=1) {
        // idx: 2, deps: [0, 1]
        let c = a.add(&b);
        assert_eq!(vec![3, 5, 4, 8, 5, 11], c.read());

        // idx: 3, deps: [2, 2]
        let d = c.relu();

        assert_eq!(vec![3, 5, 4, 8, 5, 11], d.read());

        // idx: 4, deps: [3, 1]
        let e = d.add(&b);

        if ep == 1 {
            assert_eq!(c.ptr, d.ptr);
            assert_eq!(c.ptr, e.ptr);
        }
        device.optimize_mem_graph(&device, None).unwrap();
    }
    Ok(())
}
