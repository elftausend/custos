use crate::{Device, ShallowCopy, Unit};

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

/// If the [`Shape`] does not matter for a specific device [`Buffer`](crate::Buffer), than this trait should be implemented.
/// # Safety
/// The implementor must ensure that created device [`Buffer`](crate::Buffer)s are unaffected by the generic `S` shape parameter.
pub unsafe trait IsShapeIndep: Device {}

pub trait IsShapeIndep2<T, D: Device> {
    fn to_shape<O: Shape>(self) -> D::Base<T, O>;
    fn as_shape<O: Shape>(&self) -> &D::Base<T, O>;
    fn as_shape_mut<O: Shape>(&mut self) -> &mut D::Base<T, O>;
}

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

#[cfg(test)]
mod tests {
    use core::mem::size_of;

    use crate::{Buffer, Device, Dim1, Dim2, Dim3, Shape};

    #[cfg(feature = "std")]
    fn len_of_shape<T: crate::Unit, D: Device, S: Shape>(_: &Buffer<T, D, S>) {
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
