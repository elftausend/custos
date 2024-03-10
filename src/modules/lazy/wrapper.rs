use core::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use crate::{flag::AllocFlag, HasId, HostPtr, Id, Lazy, PtrType, ShallowCopy, Shape, WrappedData};

#[derive(Debug, Default)]
pub struct LazyWrapper<Data, T> {
    pub data: Option<Data>,
    pub id: Option<Id>,
    pub _pd: PhantomData<T>,
}

impl<Mods: WrappedData> WrappedData for Lazy<Mods> {
    type Wrap<T, Base: HasId + PtrType> = LazyWrapper<Mods::Wrap<T, Base>, T>;

    #[inline]
    fn wrap_in_base<T, Base: HasId + PtrType>(&self, base: Base) -> Self::Wrap<T, Base> {
        LazyWrapper {
            data: Some(self.modules.wrap_in_base(base)),
            id: None,
            _pd: PhantomData,
        }
    }

    #[inline]
    fn wrapped_as_base<T, Base: HasId + PtrType>(wrap: &Self::Wrap<T, Base>) -> &Base {
        Mods::wrapped_as_base(wrap.data.as_ref().expect(MISSING_DATA))
    }

    #[inline]
    fn wrapped_as_base_mut<T, Base: HasId + PtrType>(wrap: &mut Self::Wrap<T, Base>) -> &mut Base {
        Mods::wrapped_as_base_mut(wrap.data.as_mut().expect(MISSING_DATA))
    }
}

impl<Data: HasId, T> HasId for LazyWrapper<Data, T> {
    #[inline]
    fn id(&self) -> crate::Id {
        match self.id {
            Some(id) => id,
            None => self.data.as_ref().unwrap().id(),
        }
    }
}

impl<Data: PtrType, T> PtrType for LazyWrapper<Data, T> {
    #[inline]
    fn size(&self) -> usize {
        match self.id {
            Some(id) => id.len,
            None => self.data.as_ref().unwrap().size(),
        }
    }

    #[inline]
    fn flag(&self) -> AllocFlag {
        self.data
            .as_ref()
            .map(|data| data.flag())
            .unwrap_or(AllocFlag::Lazy)
    }

    #[inline]
    unsafe fn set_flag(&mut self, flag: AllocFlag) {
        self.data.as_mut().unwrap().set_flag(flag)
    }
}

const MISSING_DATA: &str =
    "This lazy buffer does not contain any data. Try with a buffer.replace() call.";

impl<Data: Deref<Target = [T]>, T> Deref for LazyWrapper<Data, T> {
    type Target = Data;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.data.as_ref().expect(MISSING_DATA)
    }
}

impl<Data: DerefMut<Target = [T]>, T> DerefMut for LazyWrapper<Data, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data.as_mut().expect(MISSING_DATA)
    }
}

impl<T, Data: HostPtr<T>> HostPtr<T> for LazyWrapper<Data, T> {
    #[inline]
    fn ptr(&self) -> *const T {
        self.data.as_ref().unwrap().ptr()
    }

    #[inline]
    fn ptr_mut(&mut self) -> *mut T {
        self.data.as_mut().unwrap().ptr_mut()
    }
}

impl<Data: ShallowCopy, T> ShallowCopy for LazyWrapper<Data, T> {
    #[inline]
    unsafe fn shallow(&self) -> Self {
        LazyWrapper {
            id: self.id,
            data: self.data.as_ref().map(|data| data.shallow()),
            _pd: PhantomData,
        }
    }
}

impl<Data: crate::ConvPtr<NewT, NewS, ConvertTo = Data>, T, NewT, NewS: Shape>
    crate::ConvPtr<NewT, NewS> for LazyWrapper<Data, T>
{
    type ConvertTo = LazyWrapper<Data, NewT>;

    #[inline]
    unsafe fn convert(&self, flag: crate::flag::AllocFlag) -> Self::ConvertTo {
        LazyWrapper {
            id: self.id,
            data: self.data.as_ref().map(|data| data.convert(flag)),
            _pd: PhantomData,
        }
    }
}
