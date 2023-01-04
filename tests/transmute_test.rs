use custos::{Buffer, Dim1, Shape, CPU};

pub struct Test {}

unsafe impl Shape for Test {
    type ARR<T> = ();

    fn new<T: Copy + Default>() -> Self::ARR<T> {
        todo!()
    }
}

#[test]
fn test_transmute_of_dims() {
    let device = CPU::new();
    let mut buf = Buffer::<f32, CPU, ()>::new(&device, 100);
    //buf.flag = BufFlag::Wrapper;

    /*let x: Buffer<f32, CPU, Dim1<7>> = Buffer {
        ptr: buf.ptr,
        device: buf.device,
        node: buf.node,
    };*/

    unsafe {
        let dst: Buffer<f32, CPU, Dim1<7>> = core::mem::transmute(buf);
    }
}
