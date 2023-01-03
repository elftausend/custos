use custos::{Buffer, Dim1, CPU};

#[test]
fn test_transmute_of_dims() {
    let device = CPU::new();
    let mut buf = Buffer::<f32, CPU, ()>::new(&device, 100);
    //buf.flag = BufFlag::Wrapper;

    let x: Buffer<f32, CPU, Dim1<7>> = Buffer {
        ptr: buf.ptr,
        device: buf.device,
        node: buf.node,
    };
}
