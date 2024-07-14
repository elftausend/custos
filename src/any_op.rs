use crate::{Buffer, Buffers, Device, Downcast};

pub trait AnyOp: Sized {
    type Replicated<'a>;

    #[cfg(feature = "std")]
    fn replication_fn<B: Downcast>(
        ids: Vec<crate::Id>,
        op: impl for<'a> Fn(Self::Replicated<'a>) -> crate::Result<()> + 'static,
    ) -> Box<dyn for<'i> Fn(&'i mut Buffers<B>) -> crate::Result<()>>;
}


pub trait Replicate {
    type Replication<'r>: 'r;
}

impl<'a, T: 'static, D: Device + 'static, S: crate::Shape> Replicate
    for &crate::Buffer<'a, T, D, S>
{
    type Replication<'r> = Buffer<'r, T, D, S>;
}

impl<'a, T: 'static, D: Device + 'static, S: crate::Shape> Replicate
    for &mut crate::Buffer<'a, T, D, S>
{
    type Replication<'r> = Buffer<'r, T, D, S>;
}

impl<R: crate::HasId + Replicate> AnyOp for R {
    #[cfg(feature = "std")]
    fn replication_fn<B: Downcast>(
        ids: Vec<crate::Id>,
        op: impl for<'a> Fn(Self::Replicated<'a>) -> crate::Result<()> + 'static,
    ) -> Box<dyn Fn(&mut Buffers<B>) -> crate::Result<()>> {
        let id = ids[0];
        Box::new(move |buffers| {
            let r1 = unsafe {
                &mut *(buffers
                    .get_mut(&*id)
                    .unwrap()
                    .downcast_mut::<R::Replication<'_>>()
                    .unwrap() as *mut _)
            };
            op(r1)
        })
    }
    type Replicated<'a> = &'a mut R::Replication<'a>;
}
