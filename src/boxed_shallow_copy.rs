use crate::{AnyBuffer, Downcast, ShallowCopy};

pub trait BoxedShallowCopy: AnyBuffer {
    fn shallow_copy(&self) -> Box<dyn BoxedShallowCopy>;
    fn as_any(&self) -> &dyn AnyBuffer;
    fn as_any_mut(&mut self) -> &mut dyn AnyBuffer;
}

impl<T: AnyBuffer + ShallowCopy + 'static> BoxedShallowCopy for T {
    #[inline]
    fn shallow_copy(&self) -> Box<dyn BoxedShallowCopy> {
        Box::new(unsafe { self.shallow() })
    }

    #[inline]
    fn as_any(&self) -> &dyn AnyBuffer {
        self
    }

    #[inline]
    fn as_any_mut(&mut self) -> &mut dyn AnyBuffer {
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
    unsafe fn downcast_mut_unchecked<T>(&mut self) -> &mut T {
        self.as_any_mut().downcast_mut_unchecked()
    }

    #[inline]
    unsafe fn downcast_ref_unchecked<T>(&self) -> &T {
        self.as_any().downcast_ref_unchecked()
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
    unsafe fn downcast_mut_unchecked<T>(&mut self) -> &mut T {
        (**self).downcast_mut_unchecked()
    }

    #[inline]
    unsafe fn downcast_ref_unchecked<T>(&self) -> &T {
        (**self).downcast_ref_unchecked()
    }

    #[inline]
    fn is<T: 'static>(&self) -> bool {
        (**self).is::<T>()
    }
}
