use std::any::Any;

use crate::{Downcast, ShallowCopy};

pub trait BoxedShallowCopy {
    unsafe fn shallow_copy(&self) -> Box<dyn BoxedShallowCopy>;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: ShallowCopy + 'static> BoxedShallowCopy for T {
    #[inline]
    unsafe fn shallow_copy(&self) -> Box<dyn BoxedShallowCopy> {
        Box::new(unsafe { self.shallow() })
    }

    #[inline]
    fn as_any(&self) -> &dyn Any {
        self
    }

    #[inline]
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Downcast for dyn BoxedShallowCopy {
    #[inline]
    fn downcast_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.as_any_mut().downcast_mut()
    }

    #[inline]
    fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.as_any().downcast_ref()
    }

    #[inline]
    fn is<T: 'static>(&self) -> bool {
        self.as_any().is::<T>()
    }
}

impl<I: Downcast + ?Sized> Downcast for Box<I> {
    #[inline]
    fn downcast_mut<T: 'static>(&mut self) -> Option<&mut T> {
        (**self).downcast_mut()
    }

    #[inline]
    fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        (**self).downcast_ref()
    }

    #[inline]
    fn is<T: 'static>(&self) -> bool {
        (**self).is::<T>()
    }
}
