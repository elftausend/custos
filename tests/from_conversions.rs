use custos::prelude::*;

#[cfg(feature = "cpu")]
#[test]
fn from_slice() {
    let cpu = CPU::<Base>::new();

    let buf = Buffer::<f32, CPU>::from((&cpu, &[2.1; 10]));
    assert_eq!(buf.as_slice(), &[2.1; 10])
}
