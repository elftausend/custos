use custos::{ApplyFunction, Base, Combiner, Device, Graph, Lazy, OpenCL, Optimize, Run};

fn main() {
    let device = OpenCL::<Graph<Lazy<Base>>>::new(0).unwrap();
    // should work with any device (except nnapi)
    // let device = CPU::<Graph<Lazy<Base>>>::new();
    let buf = device.buffer([1., 2., 3., 4., 5.]);

    let out1 = device.apply_fn(&buf, |x| x.add(1.));
    let out2 = device.apply_fn(&out1, |x| x.sin());

    // this identifies redundant intermediate buffers and skips allocating them
    unsafe {
        device.optimize_mem_graph(&device, None).unwrap();
    } // allocates, now out1 data points to out2 data. The data is accessed with out2.replace()
    // this fuses all unary operations and creates fused compute kernels (for all compute kernel based devices)
    device.unary_fusing(&device, None).unwrap();

    // this executes all operations inside the lazy graph
    device.run().unwrap();

    for (input, out) in buf.read().iter().zip(out2.replace().read()) {
        assert!((out - (input + 1.).sin()).abs() < 0.01);
    }
}
