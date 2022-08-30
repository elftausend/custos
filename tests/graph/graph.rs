use custos::{range, Buffer, GraphOpt, CPU};

#[cfg(feature="opencl")]
use custos::CLDevice;

use crate::graph::AddOp;

#[test]
fn test_graph() {
    let device = CPU::new();

    // idx: 0
    let a = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));
    // idx: 1
    let b = Buffer::from((&device, [2, 3, 1, 4, 0, 5]));

    for ep in range(1) {
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
        device.optimize();
    }
}

#[cfg(feature="opencl")]
#[test]
fn test_graph_cl() -> custos::Result<()> {
    let device = CLDevice::new(0)?;

    // idx: 0
    let a = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));
    // idx: 1
    let b = Buffer::from((&device, [2, 3, 1, 4, 0, 5]));

    for ep in range(1) {
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
        device.optimize();
    }
    Ok(())
}

#[cfg(feature="cuda")]
#[test]
fn test_graph_cu() -> custos::Result<()> {
    use custos::CudaDevice;

    let device = CudaDevice::new(0)?;

    // idx: 0
    let a = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));
    // idx: 1
    let b = Buffer::from((&device, [2, 3, 1, 4, 0, 5]));

    for ep in range(1) {
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
        device.optimize();
    }
    Ok(())
}