use crate::{HasId, PtrType};

pub trait WrappedData {
    type Wrap<T, Base: HasId + PtrType>: HasId + PtrType;

    fn wrap_in_base<T, Base: HasId + PtrType>(&self, base: Base) -> Self::Wrap<T, Base>;
}
