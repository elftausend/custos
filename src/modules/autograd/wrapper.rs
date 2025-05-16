use core::{fmt::Debug, marker::PhantomData};

use crate::{
    Autograd, Device, HasId, IsBasePtr, PtrType, ShallowCopy, Shape, ToBase, ToDim, UniqueId, Unit,
    WrappedData, flag::AllocFlag,
};

// #[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ReqGradWrapper<'a, Data: HasId, T> {
    pub requires_grad: bool,
    pub data: Data,
    pub remove_id_cb: Option<Box<dyn Fn(UniqueId) + 'a>>,
    pub _pd: PhantomData<&'a T>,
}

impl<'a, Data: HasId + Debug, T> Debug for ReqGradWrapper<'a, Data, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ReqGradWrapper")
            .field("requires_grad", &self.requires_grad)
            .field("data", &self.data)
            .field("remove_id_cb", &"callback()")
            .field("_pd", &self._pd)
            .finish()
    }
}

impl<'a, Data: HasId, T> Drop for ReqGradWrapper<'a, Data, T> {
    #[inline]
    fn drop(&mut self) {
        // TODO
        // FIXME if an alloc flag None buffer goes out of scope and it has used it's gradient buffer before,
        // the gradient buffer will stay allocated
        // - deallocate directly -> however, a user storing the id maybe wants to retrieve the grad buf
        // - add to id set of potentially unused buffers

        if let Some(remove_id_cb) = &self.remove_id_cb {
            remove_id_cb(*self.id())
        }
    }
}

impl<'a, Data: HasId, T> ReqGradWrapper<'a, Data, T> {
    #[inline]
    pub fn new(data: Data, remove_id_cb: Option<Box<dyn Fn(UniqueId) + 'a>>) -> Self {
        // by default: true -> if lazy layer is (accidentally) put before autograd, all gradients will be computed instead of none.. subject to change
        ReqGradWrapper {
            requires_grad: true,
            data,
            remove_id_cb,
            _pd: PhantomData,
        }
    }
}

impl<'dev, Mods: WrappedData> WrappedData for Autograd<'dev, Mods> {
    type Wrap<'a, T: Unit, Base: IsBasePtr> = ReqGradWrapper<'a, Mods::Wrap<'a, T, Base>, T>;

    #[inline]
    fn wrap_in_base<'a, T: Unit, Base: IsBasePtr>(&'a self, base: Base) -> Self::Wrap<'a, T, Base> {
        ReqGradWrapper::new(
            self.modules.wrap_in_base(base),
            Some(Box::new(|id| {
                unsafe { (*self.grads.get()).buf_requires_grad.remove(&id) };
                unsafe { (*self.grads.get()).no_grads_pool.remove(&id) };
            })),
        )
    }

    #[inline]
    fn wrap_in_base_unbound<'a, T: Unit, Base: IsBasePtr>(
        &self,
        base: Base,
    ) -> Self::Wrap<'a, T, Base> {
        ReqGradWrapper::new(self.modules.wrap_in_base_unbound(base), None)
    }

    #[inline]
    fn wrapped_as_base<'a, 'b, T: Unit, Base: IsBasePtr>(
        wrap: &'b Self::Wrap<'a, T, Base>,
    ) -> &'b Base {
        Mods::wrapped_as_base(&wrap.data)
    }

    #[inline]
    fn wrapped_as_base_mut<'a, 'b, T: Unit, Base: IsBasePtr>(
        wrap: &'b mut Self::Wrap<'a, T, Base>,
    ) -> &'b mut Base {
        Mods::wrapped_as_base_mut(&mut wrap.data)
    }
}

impl<'a, Data: HasId, T> HasId for ReqGradWrapper<'a, Data, T> {
    #[inline]
    fn id(&self) -> crate::Id {
        self.data.id()
    }

    #[inline]
    fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    #[inline]
    fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }
}

impl<'a, Data: HasId + PtrType, T: Unit> PtrType for ReqGradWrapper<'a, Data, T> {
    #[inline]
    fn size(&self) -> usize {
        self.data.size()
    }

    #[inline]
    fn flag(&self) -> AllocFlag {
        self.data.flag()
    }

    #[inline]
    unsafe fn set_flag(&mut self, flag: AllocFlag) {
        unsafe { self.data.set_flag(flag) }
    }
}

impl<'a, Data, T> ShallowCopy for ReqGradWrapper<'a, Data, T>
where
    Data: ShallowCopy + HasId,
{
    unsafe fn shallow(&self) -> Self {
        ReqGradWrapper {
            requires_grad: self.requires_grad,
            data: unsafe { self.data.shallow() },
            remove_id_cb: None,
            _pd: PhantomData,
        }
    }
}

impl<'a, T: Unit, S: Shape, Data: ToBase<T, D, S> + HasId, T1, D: Device> ToBase<T, D, S>
    for ReqGradWrapper<'a, Data, T1>
{
    #[inline]
    fn to_base(self) -> <D as Device>::Base<T, S> {
        todo!()
        // self.data.to_base()
    }
}

impl<'a, T, Data: HasId> ToDim for ReqGradWrapper<'a, Data, T> {
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
