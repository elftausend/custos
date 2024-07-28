use crate::{Buffer, Device, Id};

#[cfg(feature = "std")]
use crate::{Buffers, Downcast};

pub trait Replicate {
    type Replication<'r>;
    type Downcast<'r>: 'r;

    #[cfg(feature = "std")]
    unsafe fn replicate_borrowed<'r, B: Downcast>(
        id: &Id,
        buffers: &'r mut Buffers<B>,
        device: Option<&'r dyn core::any::Any>,
    ) -> Option<Self::Replication<'r>>;

    unsafe fn replicate<'a>(self) -> Self::Replication<'a>;
}

pub trait Replicate2<D: Device> {
    type Replication<'r>;
    type Downcast<'r>: 'r;

    #[cfg(feature = "std")]
    unsafe fn replicate_borrowed<'r, B: Downcast>(
        id: &Id,
        buffers: &'r mut Buffers<B>,
        device: Option<&'r D>,
    ) -> Option<Self::Replication<'r>>;

    unsafe fn replicate<'a>(self) -> Self::Replication<'a>;
}

pub trait AnyOp: Sized {
    type Replicated<'a>;

    #[cfg(feature = "std")]
    fn replication_fn<B: Downcast>(
        op: impl for<'a> Fn(Self::Replicated<'a>) -> crate::Result<()> + 'static,
    ) -> Box<dyn for<'i> Fn(&[crate::Id], &'i mut Buffers<B>, &dyn core::any::Any) -> crate::Result<()>>;

    unsafe fn replication<'a>(self) -> Self::Replicated<'a>;
}

impl<'a, T: 'static, D: Device + 'static, S: crate::Shape> Replicate
    for &crate::Buffer<'a, T, D, S>
{
    type Replication<'r> = &'r Buffer<'r, T, D, S>;
    type Downcast<'r> = Buffer<'r, T, D, S>;

    #[cfg(feature = "std")]
    #[inline]
    unsafe fn replicate_borrowed<'r, B: Downcast>(
        id: &Id,
        buffers: &'r mut Buffers<B>,
        device: Option<&'r dyn core::any::Any>,
    ) -> Option<Self::Replication<'r>> {
        <&mut Buffer<T, D, S> as Replicate>::replicate_borrowed(id, buffers, device)
            .map(|buf| &*buf)
    }

    #[inline]
    unsafe fn replicate<'r>(self) -> Self::Replication<'r> {
        // TODO: this should work without this trick -> move 'own, 'dev up to the trait when something like for<'a: 'b, ...> starts to work
        // https://github.com/rust-lang/rust/issues/100013
        // look at commit "0d54d19a52979352ec59f1619a439541e08c30a0" - it was implemented like this there
        // most of the "double lifetime stuff" is still implemented at the moment
        // commit a985577299335ab00a02dc226a2e4b9d1642b8f7 introduced this line
        unsafe { core::mem::transmute::<Self, &Buffer<'r, T, D, S>>(self) }
    }
}

impl<'a, T: 'static, D: Device + 'static, S: crate::Shape> Replicate
    for &mut crate::Buffer<'a, T, D, S>
{
    type Replication<'r> = &'r mut Self::Downcast<'r>;
    type Downcast<'r> = Buffer<'r, T, D, S>;

    #[cfg(feature = "std")]
    unsafe fn replicate_borrowed<'r, B: Downcast>(
        id: &Id,
        buffers: &'r mut Buffers<B>,
        device: Option<&'r dyn core::any::Any>,
    ) -> Option<Self::Replication<'r>> {
        let replication = buffers.get_mut(id)?;
        if !replication.is::<Self::Downcast<'_>>() {
            return None;
        }
        let buf = unsafe { replication.downcast_mut_unchecked::<Self::Downcast<'r>>() };
        buf.device = device.map(|dev| dev.downcast_ref::<D>().unwrap());
        Some(buf)
    }

    #[inline]
    unsafe fn replicate<'r>(self) -> Self::Replication<'r> {
        // TODO: this should work without this trick -> move 'own, 'dev up to the trait when something like for<'a: 'b, ...> starts to work
        // https://github.com/rust-lang/rust/issues/100013
        // look at commit "0d54d19a52979352ec59f1619a439541e08c30a0" - it was implemented like this there
        // most of the "double lifetime stuff" is still implemented at the moment
        // commit a985577299335ab00a02dc226a2e4b9d1642b8f7 introduced this line
        unsafe { core::mem::transmute::<Self, &mut Buffer<'r, T, D, S>>(self) }
    }
}

impl<R: crate::HasId + Replicate> AnyOp for R {
    #[cfg(feature = "std")]
    fn replication_fn<B: Downcast>(
        op: impl for<'a> Fn(Self::Replicated<'a>) -> crate::Result<()> + 'static,
    ) -> Box<dyn Fn(&[crate::Id], &mut Buffers<B>, &dyn core::any::Any) -> crate::Result<()>> {
        use crate::DeviceError;

        Box::new(move |ids, buffers, dev| {
            let id = ids[0];
            let r1 = unsafe { R::replicate_borrowed(&id, buffers, Some(dev)) }
                .ok_or(DeviceError::InvalidLazyBuf)?;
            op(r1)
        })
    }
    type Replicated<'a> = R::Replication<'a>;

    #[inline]
    unsafe fn replication<'a>(self) -> Self::Replicated<'a> {
        self.replicate()
    }
}

/*
pub trait AnyOp2<'own, 'dev>: Sized {
    type Replicated<'a, 'b>
    where
        'b: 'a;

    #[cfg(feature = "std")]
    fn replication_fn<B: Downcast>(
        ids: Vec<crate::Id>,
        op: impl for<'a, 'b> Fn(Self::Replicated<'a, 'a>) -> crate::Result<()> + 'static,
    ) -> Box<dyn for<'i> Fn(&'i mut Buffers<B>) -> crate::Result<()>>;

    unsafe fn replication<'iown, 'idev>(self) -> Self::Replicated<'iown, 'idev>;
}

pub trait Replicate2<'uown, 'udev> {
    type Replication<'r, 'd>
    where
        'd: 'r;
    type Downcast<'r>: 'r;

    unsafe fn replicate<'own, 'dev>(self) -> Self::Replication<'own, 'dev>;

    #[cfg(feature = "std")]
    unsafe fn replicate_borrowed<'r, B: Downcast>(
        id: &Id,
        buffers: &'r mut Buffers<B>,
    ) -> Option<Self::Replication<'r, 'r>>;
}

impl<'uown, 'a, T: 'static, D: Device + 'static, S: crate::Shape> Replicate2<'uown, 'a>
    for &'uown crate::Buffer<'a, T, D, S>
{
    type Replication<'r, 'd> = &'r Buffer<'r, T, D, S> where 'd: 'r;
    type Downcast<'r> = Buffer<'r, T, D, S>;

    #[cfg(feature = "std")]
    unsafe fn replicate_borrowed<'r, B: Downcast>(
        id: &Id,
        buffers: &'r mut Buffers<B>,
    ) -> Option<Self::Replication<'r, 'r>> {
        buffers.get(id)?.downcast_ref::<Self::Downcast<'_>>()
    }

    unsafe fn replicate<'own, 'dev: 'own>(self) -> Self::Replication<'own, 'dev> {
        // TODO: this should work without this trick -> move 'own, 'dev up to the trait when something like for<'a: 'b, ...> starts to work
        // look at commit "0d54d19a52979352ec59f1619a439541e08c30a0" - it was implemented like this there
        // but than something like this happens: https://github.com/rust-lang/rust/issues/100013
        // most of the "double lifetime stuff" is still implemented at the moment
        // commit a985577299335ab00a02dc226a2e4b9d1642b8f7 introduced this line
        unsafe { core::mem::transmute::<Self, &'own Buffer<'dev, T, D, S>>(self) }
    }
}

impl<'uown, 'udev, T: 'static, D: Device + 'static, S: crate::Shape> Replicate2<'uown, 'udev>
    for &'uown mut crate::Buffer<'udev, T, D, S>
{
    type Replication<'r, 'd> = &'r mut Self::Downcast<'d> where 'd: 'r;
    type Downcast<'r> = Buffer<'r, T, D, S>;

    #[cfg(feature = "std")]
    unsafe fn replicate_borrowed<'r, B: Downcast>(
        id: &Id,
        buffers: &'r mut Buffers<B>,
    ) -> Option<Self::Replication<'r, 'r>> {
        let replication = buffers.get_mut(id)?;
        if !replication.is::<Self::Downcast<'_>>() {
            return None;
        }
        Some(unsafe { replication.downcast_mut_unchecked::<Self::Downcast<'r>>() })
    }

    unsafe fn replicate<'own, 'dev>(self) -> Self::Replication<'own, 'dev> {
        // TODO: this should work without this trick -> move 'own, 'dev up to the trait when something like for<'a: 'b, ...> starts to work
        // https://github.com/rust-lang/rust/issues/100013
        // look at commit "0d54d19a52979352ec59f1619a439541e08c30a0" - it was implemented like this there
        // most of the "double lifetime stuff" is still implemented at the moment
        // commit a985577299335ab00a02dc226a2e4b9d1642b8f7 introduced this line
        unsafe { core::mem::transmute::<Self, &'own mut Buffer<'dev, T, D, S>>(self) }
    }
}


impl<'own, 'dev, R: crate::HasId + Replicate2<'own, 'dev>> AnyOp2<'own, 'dev> for R {
    type Replicated<'a, 'b> = R::Replication<'a, 'b> where 'b: 'a;

    #[cfg(feature = "std")]
    fn replication_fn<B: Downcast>(
        ids: Vec<crate::Id>,
        op: impl for<'a> Fn(Self::Replicated<'a, 'a>) -> crate::Result<()> + 'static,
    ) -> Box<dyn Fn(&mut Buffers<B>) -> crate::Result<()>> {
        use crate::DeviceError;

        let id = ids[0];
        Box::new(move |buffers| {
            let r1 = unsafe { R::replicate_borrowed(&id, buffers) }
                .ok_or(DeviceError::InvalidLazyBuf)?;
            op(r1)
        })
    }

    #[inline]
    unsafe fn replication<'iown, 'idev: 'iown>(self) -> Self::Replicated<'iown, 'idev> {
        unsafe { self.replicate() }
    }
}*/
