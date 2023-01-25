use crate::{Device, PtrType};

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

// TODO: impl for net device
// this is used to
pub trait IsShapeIndep: Device {}
impl<D: crate::RawConv> IsShapeIndep for D {}

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

impl<T, D: crate::RawConv, I: Shape, O: Shape> ToDim<T, I, O> for D
where
    Self::Ptr<T, ()>: crate::PtrType,
{
    #[inline]
    fn to_dim(&self, ptr: Self::Ptr<T, I>) -> D::Ptr<T, O> {
        // resources are now mananged by the destructed raw pointer (prevents double free).
        let ptr = core::mem::ManuallyDrop::new(ptr);

        let raw_ptr = D::construct(&ptr, ptr.len(), ptr.flag());
        let ptr = D::destruct(&raw_ptr);

        core::mem::forget(raw_ptr);

        ptr
    }
}

/*
impl<T, D: crate::RawConv, I: IsConstDim> ToDim<T, I, ()> for D
where
    Self::Ptr<T, I>: crate::PtrType,
{
    #[inline]
    fn to_dim(&self, ptr: Self::Ptr<T, I>) -> D::Ptr<T, ()> {
        // resources are now mananged by the destructed raw pointer (prevents double free).
        let ptr = core::mem::ManuallyDrop::new(ptr);
        // TODO: mind default node!
        let raw_ptr = D::construct(&ptr, ptr.len(), Default::default());
        let (ptr, _) = D::destruct(&raw_ptr, ptr.flag());

        core::mem::forget(raw_ptr);

        ptr
    }
}*/

/*
impl<T, D: Device, S: IsConstDim> ToDim<T, S, S> for D {
    #[inline]
    fn to_dim(&self, ptr: Self::Ptr<T, S>) -> D::Ptr<T, S> {
        ptr
    }
}
*/

#[cfg(feature = "stack")]
impl<T, S: IsConstDim> ToDim<T, S, S> for crate::Stack {
    #[inline]
    fn to_dim(&self, ptr: Self::Ptr<T, S>) -> Self::Ptr<T, S> {
        ptr
    }
}

#[cfg(test)]
mod tests {
    use core::mem::size_of;

    use crate::{Buffer, Device, Dim1, Dim2, Dim3, Shape};

    fn len_of_shape<T, D: Device, S: Shape>(_: &Buffer<T, D, S>) {
        println!("S::LEN {}", S::LEN);
    }

    #[test]
    fn test_size_of_dims() {
        assert_eq!(0, size_of::<Dim1<20>>());
        assert_eq!(0, size_of::<Dim2<20, 10>>());
        assert_eq!(0, size_of::<Dim3<4134, 20, 10>>());
        assert_eq!(0, size_of::<()>());
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_transmute_of_stackless_buf() {
        use crate::{Buffer, CPU};

        let device = CPU::new();
        let buf = Buffer::<f32, CPU, Dim2<5, 5>>::new(&device, 10);

        let other_buf = unsafe {
            &*(&buf as *const Buffer<f32, CPU, Dim2<5, 5>> as *const Buffer<f32, CPU, ()>)
        };

        /*
        let other_buf = unsafe {
            core::mem::transmute::<_, &Buffer::<f32, CPU, Dim3<4,4,2>>>(&buf)
        };*/

        println!("other_buf: {:?}", other_buf.read());

        len_of_shape(&other_buf);

        println!("other_buf: {:?}", other_buf.read());

        len_of_shape(&other_buf);
    }
}
