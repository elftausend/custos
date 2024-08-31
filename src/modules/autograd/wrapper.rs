use core::marker::PhantomData;

use crate::{flag::AllocFlag, Autograd, HasId, PtrType, ShallowCopy, Shape, WrappedData};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ReqGradWrapper<Data, T> {
    pub requires_grad: bool,
    pub data: Data,
    pub _pd: PhantomData<T>,
}

impl<'dev, Mods: WrappedData> WrappedData for Autograd<'dev, Mods> {
    type Wrap<T, Base: crate::HasId + crate::PtrType> = ReqGradWrapper<Mods::Wrap<T, Base>, T>;

    #[inline]
    fn wrap_in_base<T, Base: crate::HasId + crate::PtrType>(
        &self,
        base: Base,
    ) -> Self::Wrap<T, Base> {
        ReqGradWrapper {
            // by default: true -> if lazy layer is (accidentally) put before autograd, all gradients will be computed instead of none.. subject to change
            requires_grad: true,
            data: self.modules.wrap_in_base(base),
            _pd: PhantomData,
        }
    }

    #[inline]
    fn wrapped_as_base<T, Base: crate::HasId + crate::PtrType>(
        wrap: &Self::Wrap<T, Base>,
    ) -> &Base {
        Mods::wrapped_as_base(&wrap.data)
    }

    #[inline]
    fn wrapped_as_base_mut<T, Base: crate::HasId + crate::PtrType>(
        wrap: &mut Self::Wrap<T, Base>,
    ) -> &mut Base {
        Mods::wrapped_as_base_mut(&mut wrap.data)
    }
}

impl<Data: HasId, T> HasId for ReqGradWrapper<Data, T> {
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

impl<Data: PtrType, T> PtrType for ReqGradWrapper<Data, T> {
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
        self.data.set_flag(flag)
    }
}

impl<Data, T> ShallowCopy for ReqGradWrapper<Data, T>
where
    Data: ShallowCopy,
{
    unsafe fn shallow(&self) -> Self {
        ReqGradWrapper {
            requires_grad: self.requires_grad,
            data: self.data.shallow(),
            _pd: PhantomData,
        }
    }
}
