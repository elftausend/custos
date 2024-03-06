use core::marker::PhantomData;

use crate::Resolve;

pub enum OpHint<T> {
    #[cfg(feature = "std")]
    Unary(Box<dyn Fn(Resolve<T>) -> Box<dyn crate::TwoWay<T>>>),
    None,
    PhantomData(PhantomData<T>),
}

#[cfg(feature = "std")]
pub fn unary<T, O: crate::TwoWay<T> + 'static>(op: impl Fn(Resolve<T>) -> O + 'static) -> OpHint<T> {
    let dyn_op = move |x: Resolve<T>| {
        let op: Box<dyn crate::TwoWay<T>> = Box::new(op(x));
        op
    };
    // Box::new(dyn_op)
    OpHint::Unary(Box::new(dyn_op))
}
