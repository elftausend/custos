use core::{
    any::{Any, TypeId},
    fmt,
};

use crate::{Buffer, Device, Shape};

pub trait Downcast {
    fn is<T: 'static>(&self) -> bool;
    fn downcast_mut<T: 'static>(&mut self) -> Option<&mut T>;
    unsafe fn downcast_mut_unchecked<T>(&mut self) -> &mut T;
    fn downcast_ref<T: 'static>(&self) -> Option<&T>;
    unsafe fn downcast_ref_unchecked<T>(&self) -> &T;
}

impl Downcast for dyn Any {
    #[inline]
    fn downcast_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.downcast_mut()
    }

    #[inline]
    fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.downcast_ref()
    }

    #[inline]
    unsafe fn downcast_mut_unchecked<T>(&mut self) -> &mut T {
        unsafe { &mut *(self as *mut dyn Any as *mut T) }
    }

    #[inline]
    unsafe fn downcast_ref_unchecked<T>(&self) -> &T {
        unsafe { &*(self as *const dyn Any as *const T) }
    }

    #[inline]
    fn is<T: 'static>(&self) -> bool {
        self.is::<T>()
    }
}

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

impl Downcast for dyn AnyBuffer {
    #[inline]
    fn downcast_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.downcast_mut()
    }

    #[inline]
    fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.downcast_ref()
    }

    #[inline]
    unsafe fn downcast_mut_unchecked<T>(&mut self) -> &mut T {
        self.downcast_mut_unchecked()
    }

    #[inline]
    unsafe fn downcast_ref_unchecked<T>(&self) -> &T {
        self.downcast_ref_unchecked()
    }

    #[inline]
    fn is<T: 'static>(&self) -> bool {
        self.is::<T>()
    }
}

impl fmt::Debug for dyn AnyBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Any").finish_non_exhaustive()
    }
}
impl<'a> dyn AnyBuffer + 'a {
    pub fn is<T: 'static>(&self) -> bool {
        core::any::TypeId::of::<T>() == self.type_id()
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
