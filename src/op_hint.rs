use crate::Resolve;

pub enum OpHint<T> {
    #[cfg(feature = "std")]
    Unary(Box<dyn Fn(Resolve<T>) -> Box<dyn crate::TwoWay<T>>>),
    None,
}
