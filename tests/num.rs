use custos::{
    number::{Float, Number, Two, Zero},
    prelude::One,
};
use std::{
    any::{Any, TypeId},
    ops::Neg,
};

#[test]
fn test_num() {
    let num = 4;
    let num_f64 = num.as_f64();
    assert_eq!(num_f64.type_id(), TypeId::of::<f64>());

    let zero: i32 = i32::zero();
    assert_eq!(zero, 0i32);

    let one: i32 = One::one();
    assert_eq!(one, 1i32);

    assert_eq!(num.as_usize().type_id(), TypeId::of::<usize>());

    let x: u8 = Number::from_u64(10);
    assert_eq!(x, 10u8);

    let x: u32 = Number::from_usize(16);
    assert_eq!(x, 16u32);

    let two: u16 = u16::two();
    assert_eq!(two, 2u16);
}

#[cfg_attr(miri, ignore)]
#[test]
fn test_float() {
    let x = 6f32;

    assert_eq!(x.neg(), -x);
    assert_eq!(Float::powi(&x, 2), x.powi(2));
    assert_eq!(Float::powf(&x, 2.5), x.powf(2.5));
    assert_eq!(Float::sin(&x), x.sin());
    assert_eq!(Float::sqrt(&x), x.sqrt());
    assert_eq!(Float::ln(&x), x.ln());
    assert_eq!(Float::squared(x), x.powi(2));
    assert_eq!(Float::exp(&x), x.exp());
    assert_eq!(Float::tanh(&x), x.tanh());
    assert_eq!(Float::cmp(x, 8.), Some(core::cmp::Ordering::Less));
    assert_eq!(Float::abs(&-5.), 5.);

    assert_eq!((-x).abs(), x);

    let x: f32 = Float::as_generic(0.4);
    assert_eq!(x, 0.4);
}
