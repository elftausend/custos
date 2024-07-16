use crate::{Buffer, Device, Id};

#[cfg(feature = "std")]
use crate::{Buffers, Downcast};

pub trait AnyOp: Sized {
    type Replicated<'a>;

    #[cfg(feature = "std")]
    fn replication_fn<B: Downcast>(
        ids: Vec<crate::Id>,
        op: impl for<'a> Fn(Self::Replicated<'a>) -> crate::Result<()> + 'static,
    ) -> Box<dyn for<'i> Fn(&'i mut Buffers<B>) -> crate::Result<()>>;
}

pub trait AnyOp2<'dev>: Sized {
    type Replicated<'a>;

    #[cfg(feature = "std")]
    fn replication_fn<B: Downcast>(
        ids: Vec<crate::Id>,
        op: impl for<'a> Fn(Self::Replicated<'a>) -> crate::Result<()> + 'static,
    ) -> Box<dyn for<'i> Fn(&'i mut Buffers<B>) -> crate::Result<()>>;

    fn replication(self) -> Self::Replicated<'dev>;
}

pub trait Replicate2<'dev> {
    type Replication<'r>;
    type Downcast<'r>: 'r;

    fn replicate(self) -> Self::Replication<'dev>;

    #[cfg(feature = "std")]
    unsafe fn replicate_borrowed<'r, B: Downcast>(
        id: &Id,
        buffers: &'r mut Buffers<B>,
    ) -> Option<Self::Replication<'r>>;
}

impl<'a, T: 'static, D: Device + 'static, S: crate::Shape> Replicate2<'a>
    for &'a crate::Buffer<'a, T, D, S>
{
    type Replication<'r> = &'r Buffer<'r, T, D, S>;
    type Downcast<'r> = Buffer<'r, T, D, S>;

    #[cfg(feature = "std")]
    unsafe fn replicate_borrowed<'r, B: Downcast>(
        id: &Id,
        buffers: &'r mut Buffers<B>,
    ) -> Option<Self::Replication<'r>> {
        buffers.get(id)?.downcast_ref::<Self::Downcast<'_>>()
    }
    
    fn replicate(self) -> Self::Replication<'a> {
        self
    }
}

pub trait Replicate {
    type Replication<'r>;
    type Downcast<'r>: 'r;

    #[cfg(feature = "std")]
    unsafe fn replicate<'r, B: Downcast>(
        id: &Id,
        buffers: &'r mut Buffers<B>,
    ) -> Option<Self::Replication<'r>>;
}

impl<'a, T: 'static, D: Device + 'static, S: crate::Shape> Replicate
    for &crate::Buffer<'a, T, D, S>
{
    type Replication<'r> = &'r Buffer<'r, T, D, S>;
    type Downcast<'r> = Buffer<'r, T, D, S>;

    #[cfg(feature = "std")]
    unsafe fn replicate<'r, B: Downcast>(
        id: &Id,
        buffers: &'r mut Buffers<B>,
    ) -> Option<Self::Replication<'r>> {
        buffers.get(id)?.downcast_ref::<Self::Downcast<'_>>()
    }
}

impl<'a, T: 'static, D: Device + 'static, S: crate::Shape> Replicate
    for &mut crate::Buffer<'a, T, D, S>
{
    type Replication<'r> = &'r mut Self::Downcast<'r>;
    type Downcast<'r> = Buffer<'r, T, D, S>;

    #[cfg(feature = "std")]
    unsafe fn replicate<'r, B: Downcast>(
        id: &Id,
        buffers: &'r mut Buffers<B>,
    ) -> Option<Self::Replication<'r>> {
        let replication = buffers.get_mut(id)?;
        if !replication.is::<Self::Downcast<'_>>() {
            return None;
        }
        Some(unsafe { replication.downcast_mut_unchecked::<Self::Downcast<'r>>() })
    }
}

impl<R: crate::HasId + Replicate> AnyOp for R {
    #[cfg(feature = "std")]
    fn replication_fn<B: Downcast>(
        ids: Vec<crate::Id>,
        op: impl for<'a> Fn(Self::Replicated<'a>) -> crate::Result<()> + 'static,
    ) -> Box<dyn Fn(&mut Buffers<B>) -> crate::Result<()>> {
        use crate::DeviceError;

        let id = ids[0];
        Box::new(move |buffers| {
            let r1 = unsafe { R::replicate(&id, buffers) }.ok_or(DeviceError::InvalidLazyBuf)?;
            op(r1)
        })
    }
    type Replicated<'a> = R::Replication<'a>;
}

impl<'dev, R: crate::HasId + Replicate2<'dev>> AnyOp2<'dev> for R {
    type Replicated<'a> = R::Replication<'a>;

    #[cfg(feature = "std")]
    fn replication_fn<B: Downcast>(
        ids: Vec<crate::Id>,
        op: impl for<'a> Fn(Self::Replicated<'a>) -> crate::Result<()> + 'static,
    ) -> Box<dyn Fn(&mut Buffers<B>) -> crate::Result<()>> {
        use crate::DeviceError;

        let id = ids[0];
        Box::new(move |buffers| {
            let r1 = unsafe { R::replicate_borrowed(&id, buffers) }.ok_or(DeviceError::InvalidLazyBuf)?;
            op(r1)
        })
    }
    
    fn replication(self) -> Self::Replicated<'dev> {
        self.replicate()
    }
}
