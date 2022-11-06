use custos::{Buffer, CPU};

#[test]
fn test_with_threads() {
    let device = CPU::new();

    let _buf = Buffer::from((&device, [1, 2, 3, 4]));

    std::thread::spawn(|| {
        let device = CPU::new();

        let _buf = Buffer::from((&device, [1, 2, 3, 4]));
    });
}
