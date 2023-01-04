use crate::{PtrType, Device};

pub unsafe trait Shape {
    const LEN: usize = 0;
    type ARR<T>;

    fn new<T: Copy + Default>() -> Self::ARR<T>;
}

unsafe impl Shape for () {
    type ARR<T> = ();

    fn new<T>() -> Self::ARR<T> {
        ()
    }
}

pub trait IsConstDim: Shape {}

#[derive(Clone, Copy)]
pub struct Dim1<const N: usize>;

impl<const N: usize> IsConstDim for Dim1<N> {}

unsafe impl<const N: usize> Shape for Dim1<N> {
    const LEN: usize = N;
    type ARR<T> = [T; N];

    #[inline]
    fn new<T: Copy + Default>() -> Self::ARR<T> {
        [T::default(); N]
    }
}

#[derive(Clone, Copy)]
pub struct Dim2<const B: usize, const A: usize>;

impl<const B: usize, const A: usize> IsConstDim for Dim2<B, A> {}

unsafe impl<const B: usize, const A: usize> Shape for Dim2<B, A> {
    const LEN: usize = B * A;
    type ARR<T> = [[T; A]; B];

    #[inline]
    fn new<T: Copy + Default>() -> Self::ARR<T> {
        [[T::default(); A]; B]
    }
}

pub trait MayDim2<const A: usize, const B: usize>: Shape {}

impl<const A: usize, const B: usize> MayDim2<A, B> for () {}

impl<const A: usize, const B: usize> MayDim2<A, B> for Dim2<A, B> {}

#[derive(Clone, Copy)]
pub struct Dim3<const C: usize, const B: usize, const A: usize>;

impl<const C: usize, const B: usize, const A: usize> IsConstDim for Dim3<C, B, A> {}

unsafe impl<const C: usize, const B: usize, const A: usize> Shape for Dim3<C, B, A> {
    const LEN: usize = B * A * C;
    type ARR<T> = [[[T; A]; B]; C];

    #[inline]
    fn new<T: Copy + Default>() -> Self::ARR<T> {
        [[[T::default(); A]; B]; C]
    }
}

// TODO: do not use device
pub trait ToDim<T, I: Shape, O: Shape>: crate::Device {
    fn to_dim(&self, ptr: Self::Ptr<T, I>) -> Self::Ptr<T, O>;
}

impl<T, D: crate::RawConv, O: Shape> ToDim<T, (), O> for D
where
    Self::Ptr<T, ()>: crate::PtrType,
{
    #[inline]
    fn to_dim(&self, ptr: Self::Ptr<T, ()>) -> D::Ptr<T, O> {
        // resources are now mananged by the destructed raw pointer (prevents double free).
        let ptr = core::mem::ManuallyDrop::new(ptr);
        // TODO: mind default node!
        let raw_ptr = D::construct(&ptr, ptr.len(), Default::default());
        let (ptr, _) = D::destruct(&raw_ptr, ptr.flag());

        core::mem::forget(raw_ptr);

        ptr
    }
}

impl<T, D: Device, S: IsConstDim> ToDim<T, S, S> for D {
    #[inline]
    fn to_dim(&self, ptr: Self::Ptr<T, S>) -> D::Ptr<T, S> {
        ptr
    }
}
