use core::{ops::{Deref, DerefMut}, marker::PhantomData};

use crate::{HasId, PtrType, WrappedData, Lazy};

pub struct LazyWrapper<Data, T> {
    data: Option<Data>,
    _pd: PhantomData<T>
}

impl<Data: HasId, T> HasId for LazyWrapper<Data, T> {
    #[inline]
    fn id(&self) -> crate::Id {
        self.data.as_ref().unwrap().id()
    }
}

impl<Data: PtrType, T> PtrType for LazyWrapper<Data, T> {
    #[inline]
    fn size(&self) -> usize {
        self.data.as_ref().unwrap().size()
    }

    #[inline]
    fn flag(&self) -> crate::flag::AllocFlag {
        self.data.as_ref().unwrap().flag()
    }

    #[inline]
    unsafe fn set_flag(&mut self, flag: crate::flag::AllocFlag) {
        self.data.as_mut().unwrap().set_flag(flag)
    }
}

impl<Data: Deref<Target = [T]>, T> Deref for LazyWrapper<Data, T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.data.as_ref().unwrap()
    }
}

impl<Data: DerefMut<Target = [T]>, T> DerefMut for LazyWrapper<Data, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data.as_mut().unwrap()
    }
}

impl<Mods: WrappedData> WrappedData for Lazy<Mods> {
    // type Wrap<T, Base: HasId + PtrType> = LazyWrapper<Mods::Wrap<T, Base>, T>;
    type Wrap<T, Base: HasId + PtrType> = Mods::Wrap<T, Base>;

    #[inline]
    fn wrap_in_base<T, Base: HasId + PtrType>(&self, base: Base) -> Self::Wrap<T, Base> {
        self.modules.wrap_in_base(base)
    }
}
