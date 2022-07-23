use custos::number::{Float, Number};
use std::any::{Any, TypeId};

#[test]
fn test_num() {
    let num = 4;
    let num_f64 = num.as_f64();
    assert_eq!(num_f64.type_id(), TypeId::of::<f64>());

    let one: i32 = Number::one();
    assert_eq!(one, 1i32);

    let zero: f32 = Number::zero();
    assert_eq!(zero, 0.);

    assert_eq!(num.as_usize().type_id(), TypeId::of::<usize>());

    let x: u8 = Number::from_u64(10);
    assert_eq!(x, 10u8);

    let x: u32 = Number::from_usize(16);
    assert_eq!(x, 16u32);

    let two: u16 = Number::two();
    assert_eq!(two, 2u16);
}

#[test]
fn test_float() {
    let x = 6f32;

    assert_eq!(x.negate(), -x);
    assert_eq!(Float::powi(&x, 2), x.powi(2));
    assert_eq!(Float::powf(&x, 2.5), x.powf(2.5));
    assert_eq!(Float::sin(&x), x.sin());
    assert_eq!(Float::sqrt(&x), x.sqrt());
    assert_eq!(Float::ln(&x), x.ln());
    assert_eq!(Float::squared(x), x.powi(2));
    assert_eq!(Float::exp(&x), x.exp());
    assert_eq!(Float::tanh(&x), x.tanh());
    assert_eq!(Float::comp(x, 8.), Some(core::cmp::Ordering::Less));

    assert_eq!(x.negate().abs(), x);

    let x: f32 = Float::as_generic(0.4);
    assert_eq!(x, 0.4);
}
