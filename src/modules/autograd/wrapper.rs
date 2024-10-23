use core::marker::PhantomData;

use crate::{flag::AllocFlag, Autograd, HasId, PtrType, ShallowCopy, Unit, WrappedCopy, WrappedData};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ReqGradWrapper<Data, T> {
    pub requires_grad: bool,
    pub data: Data,
    pub _pd: PhantomData<T>,
}

impl<'dev, Mods: WrappedData> WrappedData for Autograd<'dev, Mods> {
    type Wrap<'a, T: Unit, Base: crate::HasId + crate::PtrType> = ReqGradWrapper<Mods::Wrap<'a, T, Base>, T>;

    #[inline]
    fn wrap_in_base<'a, T: Unit, Base: crate::HasId + crate::PtrType>(
        &self,
        base: Base,
    ) -> Self::Wrap<'a, T, Base> {
        ReqGradWrapper {
            // by default: true -> if lazy layer is (accidentally) put before autograd, all gradients will be computed instead of none.. subject to change
            requires_grad: true,
            data: self.modules.wrap_in_base(base),
            _pd: PhantomData,
        }
    }

    #[inline]
    fn wrapped_as_base<'a, 'b, T: Unit, Base: crate::HasId + crate::PtrType>(
        wrap: &'b Self::Wrap<'a, T, Base>,
    ) -> &'b Base {
        Mods::wrapped_as_base(&wrap.data)
    }

    #[inline]
    fn wrapped_as_base_mut<'a, 'b, T: Unit, Base: crate::HasId + crate::PtrType>(
        wrap: &'b mut Self::Wrap<'a, T, Base>,
    ) -> &'b mut Base {
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

impl<Data: PtrType, T: Unit> PtrType for ReqGradWrapper<Data, T> {
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

impl<Data, T> WrappedCopy for ReqGradWrapper<Data, T>
where
    Data: WrappedCopy<Base = T>,
{
    type Base = T;

    #[inline]
    fn wrapped_copy(&self, to_wrap: Self::Base) -> Self {
        Self {
            requires_grad: self.requires_grad,
            data: self.data.wrapped_copy(to_wrap),
            _pd: PhantomData,
        }
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
