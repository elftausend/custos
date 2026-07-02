use crate::{Buffer, Device};

#[cfg(feature = "std")]
use crate::Downcast;

pub trait Replicate {
    type Replication<'r>;
    type Downcast<'r>: 'r;

    /// Replicates the argument from a type erased, deviceless buffer (stored as
    /// `Buffer<'static, ..>`, see `register_buf_copyable`). The replicated buffer
    /// stays deviceless - operations receive the device as a separate argument.
    /// # Safety
    /// The returned replication aliases the type erased buffer in `entry` with an
    /// adjusted ('static -> 'r) lifetime. No (device) reference outliving 'r may
    /// be stored in it.
    #[cfg(feature = "std")]
    unsafe fn replicate_borrowed<'r, B: Downcast>(entry: &'r mut B)
    -> Option<Self::Replication<'r>>;

    unsafe fn replicate<'a>(self) -> Self::Replication<'a>;
}

pub trait AnyOp: Sized {
    type Replicated<'a>;

    #[cfg(feature = "std")]
    fn replication_fn<D: 'static, B: Downcast>(
        op: impl for<'a> Fn(Self::Replicated<'a>, &D) -> crate::Result<()> + 'static,
    ) -> crate::OperationFn<B>;

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
        entry: &'r mut B,
    ) -> Option<Self::Replication<'r>> {
        unsafe { <&mut Buffer<T, D, S> as Replicate>::replicate_borrowed(entry).map(|buf| &*buf) }
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
        entry: &'r mut B,
    ) -> Option<Self::Replication<'r>> {
        // buffers are stored deviceless as `Buffer<'static, ..>` (see `register_buf_copyable`),
        // therefore the safe, checked downcast covers everything but the lifetime
        let buf = entry.downcast_mut::<Self::Downcast<'static>>()?;
        // the buffer stays deviceless - ops use their device parameter instead
        Some(unsafe {
            core::mem::transmute::<&'r mut Buffer<'static, T, D, S>, &'r mut Buffer<'r, T, D, S>>(
                buf,
            )
        })
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

impl AnyOp for () {
    type Replicated<'a> = ();

    #[cfg(feature = "std")]
    #[inline]
    fn replication_fn<D: 'static, B: Downcast>(
        op: impl for<'a> Fn(Self::Replicated<'a>, &D) -> crate::Result<()> + 'static,
    ) -> crate::OperationFn<B> {
        Box::new(move |_ids, _buffers, dev| op((), dev.downcast_ref().unwrap()))
    }

    #[inline]
    unsafe fn replication<'a>(self) -> Self::Replicated<'a> {
        self
    }
}

impl<R: crate::HasId + Replicate> AnyOp for R {
    #[cfg(feature = "std")]
    fn replication_fn<D: 'static, B: Downcast>(
        op: impl for<'a> Fn(Self::Replicated<'a>, &D) -> crate::Result<()> + 'static,
    ) -> crate::OperationFn<B> {
        use crate::DeviceError;

        Box::new(move |ids, buffers, dev| {
            let entry = buffers
                .get_mut(&*ids[0])
                .ok_or(DeviceError::InvalidLazyBuf)?;
            let r1 = unsafe { R::replicate_borrowed(entry) }.ok_or(DeviceError::InvalidLazyBuf)?;
            op(r1, dev.downcast_ref().unwrap())
        })
    }
    type Replicated<'a> = R::Replication<'a>;

    #[inline]
    unsafe fn replication<'a>(self) -> Self::Replicated<'a> {
        unsafe { self.replicate() }
    }
}
