use core::{any::TypeId, fmt};

use crate::{Buffer, Device, Shape};

pub trait AnyBuffer {
    fn type_id(&self) -> TypeId;
}

impl<'a, T, D, S> AnyBuffer for Buffer<'a, T, D, S>
where
    T: 'static,
    D: Device + 'static,
    S: Shape,
{
    #[inline]
    fn type_id(&self) -> TypeId {
        TypeId::of::<Buffer<T, D, S>>()
    }
}

impl fmt::Debug for dyn AnyBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Any").finish_non_exhaustive()
    }
}
impl<'a> dyn AnyBuffer + 'a {
    pub fn is<T: 'static>(&self) -> bool {
        std::any::TypeId::of::<T>() == self.type_id()
    }

    #[inline]
    pub unsafe fn downcast_mut_unchecked<T>(&mut self) -> &mut T {
        // SAFETY: caller guarantees that T is the correct type
        unsafe { &mut *(self as *mut (dyn AnyBuffer + 'a) as *mut T) }
    }

    #[inline]
    pub unsafe fn downcast_ref_unchecked<T>(&self) -> &T {
        // SAFETY: caller guarantees that T is the correct type
        unsafe { &*(self as *const (dyn AnyBuffer + 'a) as *const T) }
    }

    #[inline]
    pub fn downcast_mut<T: 'static>(&mut self) -> Option<&mut T> {
        if self.is::<T>() {
            Some(unsafe { self.downcast_mut_unchecked() })
        } else {
            None
        }
    }

    #[inline]
    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        if self.is::<T>() {
            Some(unsafe { self.downcast_ref_unchecked() })
        } else {
            None
        }
    }
}
