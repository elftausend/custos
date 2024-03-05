use crate::Resolve;

pub enum OpHint<T> {
    Unary(Box<dyn Fn(Resolve<T>) -> Box<dyn crate::TwoWay<T>>>),
    None,
}

