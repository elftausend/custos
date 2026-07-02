use core::any::Any;

pub trait Downcast {
    fn is<T: 'static>(&self) -> bool;
    fn downcast_mut<T: 'static>(&mut self) -> Option<&mut T>;
    fn downcast_ref<T: 'static>(&self) -> Option<&T>;
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
    fn is<T: 'static>(&self) -> bool {
        self.is::<T>()
    }
}
