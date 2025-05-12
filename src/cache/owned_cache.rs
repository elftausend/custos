mod fast_cache;
mod length_cache;

use core::{
    any::Any,
    cell::{Ref, RefMut},
};

pub use fast_cache::*;
pub use length_cache::*;

use super::{State, UniqueId};

pub trait Cache {
    type CachedValue: DynAnyWrapper;
    fn get_mut(&self, id: UniqueId, len: usize) -> State<RefMut<Self::CachedValue>>;
    fn get(&self, id: UniqueId, len: usize) -> State<Ref<Self::CachedValue>>;
    fn insert(&self, id: UniqueId, len: usize, data: Self::CachedValue);
}

pub trait AsAny {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: 'static> AsAny for Box<T> {
    #[inline]
    fn as_any(&self) -> &dyn Any {
        &**self
    }

    #[inline]
    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut **self
    }
}

impl AsAny for Box<dyn Any> {
    #[inline]
    fn as_any(&self) -> &dyn Any {
        &**self
    }

    #[inline]
    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut **self
    }
}

#[cfg(feature = "std")]
impl AsAny for std::rc::Rc<dyn Any> {
    #[inline]
    fn as_any(&self) -> &dyn Any {
        &**self
    }

    #[inline]
    fn as_any_mut(&mut self) -> &mut dyn Any {
        unimplemented!()
    }
}

#[cfg(feature = "std")]
impl<T: 'static> AsAny for std::rc::Rc<T> {
    #[inline]
    fn as_any(&self) -> &dyn Any {
        &**self
    }

    #[inline]
    fn as_any_mut(&mut self) -> &mut dyn Any {
        unimplemented!()
    }
}

#[cfg(feature = "std")]
impl<T: 'static> AsAny for std::sync::Arc<T> {
    #[inline]
    fn as_any(&self) -> &dyn Any {
        &**self
    }

    #[inline]
    fn as_any_mut(&mut self) -> &mut dyn Any {
        unimplemented!()
    }
}

#[cfg(feature = "std")]
impl AsAny for std::sync::Arc<dyn Any> {
    #[inline]
    fn as_any(&self) -> &dyn Any {
        &**self
    }

    #[inline]
    fn as_any_mut(&mut self) -> &mut dyn Any {
        unimplemented!()
    }
}

pub trait DynAnyWrapper: AsAny {
    fn new<T: 'static>(data: T) -> Self;
}

impl DynAnyWrapper for Box<dyn Any> {
    fn new<ToWrap: 'static>(data: ToWrap) -> Self {
        Box::new(data)
    }
}

#[cfg(feature = "std")]
impl DynAnyWrapper for std::sync::Arc<dyn Any> {
    fn new<ToWrap: 'static>(data: ToWrap) -> Self {
        std::sync::Arc::new(data)
    }
}
