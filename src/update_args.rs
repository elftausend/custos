use crate::UniqueId;

#[cfg(feature = "std")]
use crate::Buffers;

/// A dummy trait for no-std context. [`UpdateArgs`] requires standard library code.
#[cfg(not(feature = "std"))]
pub trait UpdateArgs {}

#[cfg(feature = "std")]
pub trait UpdateArgs {
    #[cfg(feature = "std")]
    fn update_args<B: AsAny>(
        &mut self,
        ids: &[Option<UniqueId>],
        buffers: &mut Buffers<B>,
    ) -> crate::Result<()>;
}

/// A dummy trait for no-std context. [`UpdateArg`] requires standard library code.
#[cfg(not(feature = "std"))]
pub trait UpdateArg {}

#[cfg(feature = "std")]
pub trait UpdateArg {
    fn update_arg<B: AsAny>(
        to_update: &mut Self,
        id: Option<UniqueId>,
        buffers: &mut Buffers<B>,
    ) -> crate::Result<()>;
}

#[cfg(feature = "std")]
impl<T: UpdateArg> UpdateArgs for T {
    fn update_args<B: AsAny>(
        &mut self,
        ids: &[Option<UniqueId>],
        buffers: &mut crate::Buffers<B>,
    ) -> crate::Result<()> {
        T::update_arg(self, ids[0], buffers)
    }
}

#[cfg(feature = "std")]
pub trait UpdateArgsDynable<B> {
    fn update_args_dynable(
        &mut self,
        ids: &[Option<UniqueId>],
        buffers: &mut Buffers<B>,
    ) -> crate::Result<()>;
}

#[cfg(feature = "std")]
impl<A: UpdateArgs, T: AsAny> UpdateArgsDynable<T> for A {
    #[inline]
    fn update_args_dynable(
        &mut self,
        ids: &[Option<UniqueId>],
        buffers: &mut Buffers<T>,
    ) -> crate::Result<()> {
        self.update_args(ids, buffers)
    }
}

pub trait AsAny {
    fn as_any(&self) -> *const ();
    fn as_any_mut(&mut self) -> *mut ();
}
