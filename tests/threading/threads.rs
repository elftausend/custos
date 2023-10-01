use custos::{Buffer, CPU};

#[cfg(feature = "cpu")]
#[test]
fn test_with_threads() {
    use custos::Base;

    let device = CPU::<Base>::new();

    //let buf = Buffer::from((&device, [1, 2, 3, 4]));
    let buf = Buffer::<f32>::deviceless(&device, 10);

    let vec: Vec<f64> = vec![1., 2., 3.];

    let a = std::thread::spawn(move || {
        let device = CPU::<Base>::new();

        let _b = &buf;
        let _a = &vec;

        let _buf = Buffer::from((&device, [1, 2, 3, 4]));
    });
    a.join().unwrap();
}
