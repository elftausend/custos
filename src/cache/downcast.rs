use core::any::Any;

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
