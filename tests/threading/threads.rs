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

#[cfg(feature = "cpu")]
#[ignore = "compile time"]
#[cfg_attr(miri, ignore)]
#[test]
fn test_with_threads_static() {
    use custos::Base;

    // use this device instead: compilation error! (as expected)
    // let device = &*Box::leak(Box::new(CPU::<custos::Cached<Base>>::new()));
    let device = &*Box::leak(Box::new(CPU::<Base>::new()));

    let buf = Buffer::<f32, _>::new(device, 10);

    let a = std::thread::spawn(move || {
        let _b = buf;

        let _buf = Buffer::from((device, [1, 2, 3, 4]));
    });
    a.join().unwrap();
}
