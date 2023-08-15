use custos::{Buffer, CPU, Base};

fn main() {
    let device = CPU::<Base>::new();

    let a = Buffer::from((&device, [5, 7, 2, 10]));
    assert_eq!(a.read(), vec![5, 7, 2, 10])
}
