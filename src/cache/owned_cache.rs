mod fast_cache;
mod length_cache;

use core::{any::Any, cell::{Ref, RefMut}};

pub use fast_cache::*;
pub use length_cache::*;

use super::{State, UniqueId};

pub trait Cache<T: DynAnyWrapper> {
    fn get_mut(&self, id: UniqueId, len: usize) -> State<RefMut<T>>;
    fn get(&self, id: UniqueId, len: usize) -> State<Ref<T>>;
    fn insert(&self, id: UniqueId, len: usize, data: T);
}

pub trait AsAny {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl AsAny for Box<dyn Any> {
    #[inline]
    fn as_any(&self) -> &dyn Any {
        self
    }

    #[inline] 
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(feature="std")]
impl AsAny for std::rc::Rc<dyn Any> {
    #[inline]
    fn as_any(&self) -> &dyn Any {
        self
    }

    #[inline] 
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(feature="std")]
impl AsAny for std::sync::Arc<dyn Any> {
    #[inline]
    fn as_any(&self) -> &dyn Any {
        self
    }

    #[inline] 
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub trait DynAnyWrapper {
    type Wrapped: AsAny;
    fn new<T: 'static>(data: T) -> Self::Wrapped;
}

impl<T> DynAnyWrapper for Box<T> {
    type Wrapped = Box<dyn Any>;

    fn new<ToWrap: 'static>(data: ToWrap) -> Self::Wrapped {
        Box::new(data)
    }
}

impl DynAnyWrapper for Box<dyn Any> {
    type Wrapped = Box<dyn Any>;

    fn new<ToWrap: 'static>(data: ToWrap) -> Self::Wrapped {
        Box::new(data)
    }
}

#[cfg(feature = "std")]
impl<T> DynAnyWrapper for std::sync::Arc<T> {
    type Wrapped = std::sync::Arc<(dyn Any + 'static)>;

    fn new<ToWrap: 'static>(data: ToWrap) -> Self::Wrapped {
        std::sync::Arc::new(data)
    }
}

#[cfg(feature = "std")]
impl DynAnyWrapper for std::sync::Arc<dyn Any> {
    type Wrapped = std::sync::Arc<(dyn Any + 'static)>;

    fn new<ToWrap: 'static>(data: ToWrap) -> Self::Wrapped {
        std::sync::Arc::new(data)
    }
}

