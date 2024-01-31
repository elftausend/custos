use core::any::Any;

use crate::{AsAny, ShallowCopy};

pub trait BoxedShallowCopy: 'static {
    fn shallow_copy(&self) -> Box<dyn BoxedShallowCopy>;
}

impl<T: ShallowCopy + 'static> BoxedShallowCopy for T {
    #[inline]
    fn shallow_copy(&self) -> Box<dyn BoxedShallowCopy> {
        Box::new(unsafe { self.shallow() })
    }
}

impl AsAny for Box<dyn BoxedShallowCopy> {
    #[inline]
    fn as_any(&self) -> *const () {
        let data = &**self;
        data as *const _ as *const ()
    }

    #[inline]
    fn as_any_mut(&mut self) -> *mut () {
        let data = &mut **self;
        data as *mut _ as *mut ()
    }
}

impl AsAny for Box<dyn Any> {
    #[inline]
    fn as_any(&self) -> *const () {
        (&**self) as *const _ as *const ()
    }

    #[inline]
    fn as_any_mut(&mut self) -> *mut () {
        (&mut **self) as *mut _ as *mut ()
    }
}
