use custos::{CPU, AsDev, Buffer};

fn main() {
    let device = CPU::new().select();

    let a = Buffer::from((&device, [5, 7, 2, 10,]));
    assert_eq!(a.read(), vec![5, 7, 2, 10])
}