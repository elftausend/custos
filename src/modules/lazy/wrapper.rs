mod maybe_data;
pub use maybe_data::MaybeData;

use core::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use crate::{
    flag::AllocFlag, Device, HasId, HostPtr, IsBasePtr, Lazy, PtrType, ShallowCopy, Shape, ToBase,
    ToDim, UniqueId, Unit, WrappedData,
};

use super::unregister_buf_copyable;

#[derive(Default)]
pub struct LazyWrapper<'a, Data: HasId, T> {
    pub maybe_data: MaybeData<Data>,
    pub remove_id_cb: Option<Box<dyn Fn(UniqueId) + 'a>>,
    pub _pd: PhantomData<&'a T>,
}

impl<'a, Data: HasId, T> Drop for LazyWrapper<'a, Data, T> {
    #[inline]
    fn drop(&mut self) {
        if let Some(remove_id_cb) = &self.remove_id_cb {
            remove_id_cb(*self.id())
        }
    }
}

impl<'a, Data: std::fmt::Debug + HasId, T> std::fmt::Debug for LazyWrapper<'a, Data, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LazyWrapper")
            .field("maybe_data", &self.maybe_data)
            .field("remove_id_cb", &"callback()")
            .field("_pd", &self._pd)
            .finish()
    }
}

impl<T2, Mods: WrappedData> WrappedData for Lazy<'_, Mods, T2> {
    type Wrap<'a, T: Unit, Base: IsBasePtr> = LazyWrapper<'a, Mods::Wrap<'a, T, Base>, T>;

    #[inline]
    fn wrap_in_base<'a, T: Unit, Base: IsBasePtr>(&'a self, base: Base) -> Self::Wrap<'a, T, Base> {
        LazyWrapper {
            maybe_data: MaybeData::Data(self.modules.wrap_in_base(base)),
            remove_id_cb: Some(Box::new(|id| {
                unregister_buf_copyable(&mut self.buffers.borrow_mut(), id)
            })),
            _pd: PhantomData,
        }
    }

    #[inline]
    fn wrap_in_base_unbound<'a, T: Unit, Base: IsBasePtr>(
        &self,
        base: Base,
    ) -> Self::Wrap<'a, T, Base> {
        LazyWrapper {
            maybe_data: MaybeData::Data(self.modules.wrap_in_base_unbound(base)),
            remove_id_cb: None,
            _pd: PhantomData,
        }
    }

    #[inline]
    fn wrapped_as_base<'a, 'b, T: Unit, Base: IsBasePtr>(
        wrap: &'b Self::Wrap<'a, T, Base>,
    ) -> &'b Base {
        Mods::wrapped_as_base(wrap.maybe_data.data().expect(MISSING_DATA))
    }

    #[inline]
    fn wrapped_as_base_mut<'a, 'b, T: Unit, Base: IsBasePtr>(
        wrap: &'b mut Self::Wrap<'a, T, Base>,
    ) -> &'b mut Base {
        Mods::wrapped_as_base_mut(wrap.maybe_data.data_mut().expect(MISSING_DATA))
    }
}

impl<'a, Data: HasId, T> HasId for LazyWrapper<'a, Data, T> {
    #[inline]
    fn id(&self) -> crate::Id {
        match self.maybe_data {
            MaybeData::Data(ref data) => data.id(),
            MaybeData::Id(id) => id,
            MaybeData::None => unimplemented!(),
        }
    }
}

impl<'a, Data: PtrType + HasId, T: Unit> PtrType for LazyWrapper<'a, Data, T> {
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

impl<'a, Data: HasId + Deref<Target = [T]>, T> Deref for LazyWrapper<'a, Data, T> {
    type Target = Data;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.maybe_data.data().expect(MISSING_DATA)
    }
}

impl<'a, Data: HasId + DerefMut<Target = [T]>, T> DerefMut for LazyWrapper<'a, Data, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.maybe_data.data_mut().expect(MISSING_DATA)
    }
}

impl<'a, T: Unit, Data: HasId + HostPtr<T>> HostPtr<T> for LazyWrapper<'a, Data, T> {
    #[inline]
    fn ptr(&self) -> *const T {
        self.maybe_data.data().unwrap().ptr()
    }

    #[inline]
    fn ptr_mut(&mut self) -> *mut T {
        self.maybe_data.data_mut().unwrap().ptr_mut()
    }
}

impl<'a, Data: HasId + ShallowCopy, T> ShallowCopy for LazyWrapper<'a, Data, T> {
    #[inline]
    unsafe fn shallow(&self) -> Self {
        LazyWrapper {
            maybe_data: match &self.maybe_data {
                MaybeData::Data(data) => MaybeData::Data(data.shallow()),
                MaybeData::Id(id) => MaybeData::Id(*id),
                MaybeData::None => unimplemented!(),
            },
            remove_id_cb: None,
            _pd: PhantomData,
        }
    }
}

impl<'a, T: Unit, S: Shape, Data: HasId + ToBase<T, D, S>, T1, D: Device> ToBase<T, D, S>
    for LazyWrapper<'a, Data, T1>
{
    #[inline]
    fn to_base(self) -> <D as Device>::Base<T, S> {
        todo!()
        // match self.maybe_data {
        //     MaybeData::Data(data) => data.to_base(),
        //     MaybeData::Id(_id) => unimplemented!("Cannot convert id wrapper to base"),
        //     MaybeData::None => unimplemented!("Cannot convert nothin to base"),
        // }
    }
}

impl<'a, T, Data: HasId> ToDim for LazyWrapper<'a, Data, T> {
    type Out = Self;

    #[inline]
    fn to_dim(self) -> Self::Out {
        self
    }

    #[inline]
    fn as_dim(&self) -> &Self::Out {
        self
    }
}
