use crate::{Device, ShallowCopy};

/// Determines the shape of a [`Buffer`](crate::Buffer).
/// `Shape` is used to get the size and ND-Array for a stack allocated `Buffer`.
pub trait Shape: 'static {
    /// The count of elements that fit into the shape.
    const LEN: usize = 0;

    /// The type of the ND-Array.
    type ARR<T>;

    /// Creates a new ND-Array with the default value of `T`.
    fn new<T: Copy + Default>() -> Self::ARR<T>;

    /// Returns the dimension of the Shape as a vector.
    /// # Example
    /// ```
    /// use custos::{Dim2, Shape};
    ///
    /// assert_eq!(Dim2::<1, 2>::dims(), vec![1, 2])
    /// ```
    #[cfg(feature = "std")]
    fn dims() -> Vec<usize>;
}

impl Shape for () {
    type ARR<T> = ();

    #[inline]
    fn new<T>() -> Self::ARR<T> {}

    #[inline]
    #[cfg(feature = "std")]
    fn dims() -> Vec<usize> {
        vec![]
    }
}

// TODO: impl for net device
// this is used to
/// If the [`Shape`] does not matter for a specific device [`Buffer`](crate::Buffer), than this trait should be implemented.
pub unsafe trait IsShapeIndep: Device {}

/// If the [`Shape`] is provides a fixed size, than this trait should be implemented.
/// Forgot how this is useful.
pub trait IsConstDim: Shape {}

/// A 1D shape.
#[derive(Clone, Copy)]
pub struct Dim1<const N: usize>;

impl<const N: usize> IsConstDim for Dim1<N> {}

impl<const N: usize> Shape for Dim1<N> {
    const LEN: usize = N;
    type ARR<T> = [T; N];

    #[inline]
    fn new<T: Copy + Default>() -> Self::ARR<T> {
        [T::default(); N]
    }

    #[inline]
    #[cfg(feature = "std")]
    fn dims() -> Vec<usize> {
        vec![N]
    }
}

/// A 2D shape.
#[derive(Clone, Copy)]
pub struct Dim2<const B: usize, const A: usize>;

impl<const B: usize, const A: usize> IsConstDim for Dim2<B, A> {}

impl<const B: usize, const A: usize> Shape for Dim2<B, A> {
    const LEN: usize = B * A;
    type ARR<T> = [[T; A]; B];

    #[inline]
    fn new<T: Copy + Default>() -> Self::ARR<T> {
        [[T::default(); A]; B]
    }

    #[inline]
    #[cfg(feature = "std")]
    fn dims() -> Vec<usize> {
        vec![B, A]
    }
}

/// The shape may be 2D or ().
pub trait MayDim2<const A: usize, const B: usize>: Shape {}

impl<const A: usize, const B: usize> MayDim2<A, B> for () {}

impl<const A: usize, const B: usize> MayDim2<A, B> for Dim2<A, B> {}

/// A 3D shape.
#[derive(Clone, Copy)]
pub struct Dim3<const C: usize, const B: usize, const A: usize>;

impl<const C: usize, const B: usize, const A: usize> IsConstDim for Dim3<C, B, A> {}

impl<const C: usize, const B: usize, const A: usize> Shape for Dim3<C, B, A> {
    const LEN: usize = B * A * C;
    type ARR<T> = [[[T; A]; B]; C];

    #[inline]
    fn new<T: Copy + Default>() -> Self::ARR<T> {
        [[[T::default(); A]; B]; C]
    }

    #[inline]
    #[cfg(feature = "std")]
    fn dims() -> Vec<usize> {
        vec![C, B, A]
    }
}

// TODO: do not use device
/// Converts a pointer to a different [`Shape`].
pub trait ToDim<T, I: Shape, O: Shape>: crate::Device {
    /// Converts a pointer to a different [`Shape`].
    /// This is only possible for [`Buffer`](crate::Buffer)s that are not allocated on the stack.
    fn to_dim(&self, ptr: Self::Data<T, I>) -> Self::Data<T, O>;
}

#[cfg(feature = "std")]
impl<T, D, I, O> ToDim<T, I, O> for D
where
    D::Data<T, O>: crate::PtrType + ShallowCopy,
    D: IsShapeIndep + Device,
    I: Shape,
    O: Shape,
{
    #[inline]
    fn to_dim(&self, ptr: Self::Data<T, I>) -> D::Data<T, O> {
        // resources are now mananged by the destructed raw pointer (prevents double free).
        // could set alloc flag as well
        let ptr = core::mem::ManuallyDrop::new(ptr);

        let shape_changed = unsafe { &*(&*ptr as *const D::Data<T, I> as *const D::Data<T, O>) };
        unsafe { shape_changed.shallow() }
    }
}

#[cfg(feature = "stack")]
impl<T, S: IsConstDim> ToDim<T, S, S> for crate::Stack {
    #[inline]
    fn to_dim(&self, ptr: Self::Data<T, S>) -> Self::Data<T, S> {
        ptr
    }
}

#[cfg(test)]
mod tests {
    use core::mem::size_of;

    use crate::{Buffer, Device, Dim1, Dim2, Dim3, Shape};

    #[cfg(feature = "std")]
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
        use crate::{Base, Buffer, CPU};

        let device = CPU::<Base>::new();
        let buf = Buffer::<f32, CPU, Dim2<5, 5>>::new(&device, 10);

        let other_buf = unsafe {
            &*(&buf as *const Buffer<f32, CPU, Dim2<5, 5>> as *const Buffer<f32, CPU, ()>)
        };

        /*
        let other_buf = unsafe {
            core::mem::transmute::<_, &Buffer::<f32, CPU, Dim3<4,4,2>>>(&buf)
        };*/

        println!("other_buf: {:?}", other_buf.read());

        len_of_shape(other_buf);

        println!("other_buf: {:?}", other_buf.read());

        len_of_shape(other_buf);
    }
}
