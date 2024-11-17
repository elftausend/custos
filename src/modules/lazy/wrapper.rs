mod maybe_data;
pub use maybe_data::MaybeData;

use core::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use crate::{
    flag::AllocFlag, HasId, HostPtr, IsBasePtr, Lazy, PtrType, ShallowCopy, Unit, WrappedCopy, WrappedData
};

#[derive(Debug, Default)]
pub struct LazyWrapper<Data, T> {
    pub maybe_data: MaybeData<Data>,
    pub _pd: PhantomData<T>,
}

impl<T2, Mods: WrappedData> WrappedData for Lazy<'_, Mods, T2> {
    type Wrap<'a, T: Unit, Base: IsBasePtr> = LazyWrapper<Mods::Wrap<'a, T, Base>, T>;

    #[inline]
    fn wrap_in_base<'a, T: Unit, Base: IsBasePtr>(&self, base: Base) -> Self::Wrap<'a, T, Base> {
        LazyWrapper {
            maybe_data: MaybeData::Data(self.modules.wrap_in_base(base)),
            _pd: PhantomData,
        }
    }

    #[inline]
    fn wrapped_as_base<'a, 'b, T: Unit, Base: IsBasePtr>(wrap: &'b Self::Wrap<'a, T, Base>) -> &'b Base {
        Mods::wrapped_as_base(wrap.maybe_data.data().expect(MISSING_DATA))
    }

    #[inline]
    fn wrapped_as_base_mut<'a, 'b, T: Unit, Base: IsBasePtr>(
        wrap: &'b mut Self::Wrap<'a, T, Base>,
    ) -> &'b mut Base {
        Mods::wrapped_as_base_mut(wrap.maybe_data.data_mut().expect(MISSING_DATA))
    }
}

impl<Data: HasId, T> HasId for LazyWrapper<Data, T> {
    #[inline]
    fn id(&self) -> crate::Id {
        match self.maybe_data {
            MaybeData::Data(ref data) => data.id(),
            MaybeData::Id(id) => id,
            MaybeData::None => unimplemented!(),
        }
    }
}

impl<Data: PtrType, T: Unit> PtrType for LazyWrapper<Data, T> {
    #[inline]
    fn size(&self) -> usize {
        match self.maybe_data {
            MaybeData::Data(ref data) => data.size(),
            MaybeData::Id(id) => id.len,
            MaybeData::None => unimplemented!(),
        }
    }

    #[inline]
    fn flag(&self) -> AllocFlag {
        self.maybe_data
            .data()
            .map(|data| data.flag())
            .unwrap_or(AllocFlag::Lazy)
    }

    #[inline]
    unsafe fn set_flag(&mut self, flag: AllocFlag) {
        self.maybe_data.data_mut().unwrap().set_flag(flag)
    }
}

const MISSING_DATA: &str =
    "This lazy buffer does not contain any data. Try with a buffer.replace() call.";

impl<Data: Deref<Target = [T]>, T> Deref for LazyWrapper<Data, T> {
    type Target = Data;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.maybe_data.data().expect(MISSING_DATA)
    }
}

impl<Data: DerefMut<Target = [T]>, T> DerefMut for LazyWrapper<Data, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.maybe_data.data_mut().expect(MISSING_DATA)
    }
}

impl<T: Unit, Data: HostPtr<T>> HostPtr<T> for LazyWrapper<Data, T> {
    #[inline]
    fn ptr(&self) -> *const T {
        self.maybe_data.data().unwrap().ptr()
    }

    #[inline]
    fn ptr_mut(&mut self) -> *mut T {
        self.maybe_data.data_mut().unwrap().ptr_mut()
    }
}

impl<Data, T> WrappedCopy for LazyWrapper<Data, T>
where
    Data: WrappedCopy<Base = T>,
{
    type Base = T;

    fn wrapped_copy(&self, to_wrap: Self::Base) -> Self {
        LazyWrapper {
            maybe_data: match &self.maybe_data {
                MaybeData::Data(data) => MaybeData::Data(data.wrapped_copy(to_wrap)),
                MaybeData::Id(id) => MaybeData::Id(*id),
                MaybeData::None => unimplemented!(),
            },
            _pd: PhantomData,
        }
    }
}

impl<Data: ShallowCopy, T> ShallowCopy for LazyWrapper<Data, T> {
    #[inline]
    unsafe fn shallow(&self) -> Self {
        LazyWrapper {
            maybe_data: match &self.maybe_data {
                MaybeData::Data(data) => MaybeData::Data(data.shallow()),
                MaybeData::Id(id) => MaybeData::Id(*id),
                MaybeData::None => unimplemented!(),
            },
            _pd: PhantomData,
        }
    }
}
