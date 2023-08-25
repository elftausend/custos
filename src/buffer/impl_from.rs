use core::ops::Range;

use crate::{
    number::Number, shape::Shape, Alloc, Buffer, Device, OnDropBuffer, OnNewBuffer, Retriever,
};

#[cfg(feature = "cpu")]
use crate::{WriteBuf, CPU};

impl<'a, T, D, const N: usize> From<(&'a D, [T; N])> for Buffer<'a, T, D>
where
    T: Clone,
    // TODO: IsShapeIndep ... find way to include Stack
    D: Alloc<T> + OnNewBuffer<T, D, ()>,
{
    #[inline]
    fn from((device, array): (&'a D, [T; N])) -> Self {
        Buffer::from_slice(device, &array)
    }
}

impl<'a, T, D> From<(&'a D, usize)> for Buffer<'a, T, D>
where
    D: Alloc<T> + OnNewBuffer<T, D, ()>,
{
    #[inline]
    fn from((device, len): (&'a D, usize)) -> Self {
        Buffer::new(device, len)
    }
}

/*impl<'a, T, D> Buffer<'a, T, D>
where
    T: Clone,
    D: Alloc<T>+ OnNewBuffer<T, D, ()>
{
    #[inline]
    pub fn from_iter<I: IntoIterator<Item = T>>(device: &'a D, iter: I) -> Self {
        Buffer::from_vec(&device, iter.into_iter().collect())
    }
}*/

#[cfg(not(feature = "no-std"))]
impl<'a, T, D> From<(&'a D, Range<usize>)> for Buffer<'a, T, D>
where
    T: Number,
    D: Alloc<T> + OnNewBuffer<T, D, ()>,
{
    #[inline]
    fn from((device, range): (&'a D, Range<usize>)) -> Self {
        Buffer::from_vec(device, range.map(|x| T::from_usize(x)).collect())
    }
}

// A more general version of the above, find way to make it work
/*#[cfg(not(feature = "no-std"))]
impl<'a, T, D, I> From<(&'a D, I)> for Buffer<'a, T, D>
where
    T: Number,
    D: Alloc<T>+ OnNewBuffer<T, D, ()>,
    I: IntoIterator<Item = T>,
{
    #[inline]
    fn from((device, range): (&'a D, Range<usize>)) -> Self {
        Buffer::from_vec(device, range.map(|x| T::from_usize(x)).collect())
    }
}*/

/*impl<'a, T, D, const N: usize> From<(&'a D, [T; N])> for Buffer<'a, T, D>
where
    T: Clone,
    D: Alloc<T>+ OnNewBuffer<T, D, ()> + IsShapeIndep,
{
    fn from((device, array): (&'a D, [T; N])) -> Self {
        Buffer {
            // TODO: with_array()
            ptr: device.with_slice(&array),
            device: Some(device),
            //node: device.graph().add_leaf(len),
            ident: Ident::new_bumped(array.len()),
        }
    }
}*/

impl<'a, T, D, const N: usize> From<(&'a D, &[T; N])> for Buffer<'a, T, D>
where
    T: Clone,
    // TODO: IsShapeIndep ... find way to include Stack
    D: Alloc<T> + OnNewBuffer<T, D, ()>,
{
    #[inline]
    fn from((device, array): (&'a D, &[T; N])) -> Self {
        Buffer::from_slice(device, array)
    }
}

/*impl<'a, T, D, const N: usize> From<(&'a D, &[T; N])> for Buffer<'a, T, D>
where
    T: Clone,
    D: Alloc<T>+ OnNewBuffer<T, D, ()> + IsShapeIndep,
{
    fn from((device, array): (&'a D, &[T; N])) -> Self {
        Buffer {
            // TODO: with_array()
            ptr: device.with_slice(array),
            device: Some(device),
            //node: device.graph().add_leaf(len),
            ident: Ident::new_bumped(array.len()),
        }
    }
}*/

impl<'a, T, D, S: Shape> From<(&'a D, &[T])> for Buffer<'a, T, D, S>
where
    T: Clone,
    // TODO: IsShapeIndep ... find way to include Stack
    D: Alloc<T> + OnNewBuffer<T, D, S>,
{
    #[inline]
    fn from((device, slice): (&'a D, &[T])) -> Self {
        Buffer::from_slice(device, slice)
    }
}

/*impl<'a, T, D, S: Shape> From<(&'a D, &[T])> for Buffer<'a, T, D, S>
where
    T: Clone,
    D: Alloc<T>+ OnNewBuffer<T, D, ()> + IsShapeIndep,
{
    fn from((device, slice): (&'a D, &[T])) -> Self {
        Buffer {
            ptr: device.with_slice(slice),
            device: Some(device),
            //node: device.graph().add_leaf(len),
            ident: Ident::new_bumped(slice.len()),
        }
    }
}*/

#[cfg(not(feature = "no-std"))]
impl<'a, T, D, S: Shape> From<(&'a D, Vec<T>)> for Buffer<'a, T, D, S>
where
    T: Clone,
    // TODO: IsShapeIndep ... find way to include Stack
    D: Alloc<T> + OnNewBuffer<T, D, S>,
{
    #[inline]
    fn from((device, vec): (&'a D, Vec<T>)) -> Self {
        Buffer::from_vec(device, vec)
    }
}

#[cfg(not(feature = "no-std"))]
impl<'a, T, D, S: Shape> From<(&'a D, &Vec<T>)> for Buffer<'a, T, D, S>
where
    T: Clone,
    // TODO: IsShapeIndep ... find way to include Stack
    D: Alloc<T> + OnNewBuffer<T, D, S>,
{
    #[inline]
    fn from((device, vec): (&'a D, &Vec<T>)) -> Self {
        Buffer::from_slice(device, vec)
    }
}

#[cfg(feature = "cpu")]
impl<'a, 'b, Mods: OnDropBuffer, T, S, D> From<(&'a D, Buffer<'b, T, CPU<Mods>, S>)>
    for Buffer<'a, T, D, S>
where
    T: 'static,
    S: Shape,
    D: WriteBuf<T, S> + Device + Retriever<T>,
{
    fn from((device, buf): (&'a D, Buffer<'b, T, CPU<Mods>, S>)) -> Self {
        let mut out = device.retrieve(buf.len(), &buf);
        device.write(&mut out, &buf);
        out
    }
}

#[cfg(test)]
mod tests {

    #[cfg(feature = "cpu")]
    #[test]
    fn test_buf_device_conversion_cpu() {
        use crate::{Base, Buffer, Read, CPU};

        let device = CPU::<Base>::new();

        let cpu = CPU::<Base>::new();
        let cpu_buf = Buffer::from((&cpu, [1, 2, 4, 5]));

        let out = Buffer::from((&device, cpu_buf));
        assert_eq!(device.read(&out), [1, 2, 4, 5]);
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_buf_device_conversion_cl() -> crate::Result<()> {
        use crate::{Base, Buffer, OpenCL, Read, CPU, opencl::chosen_cl_idx};

        let device = OpenCL::<Base>::new(chosen_cl_idx())?;
        println!("name: {:?}", device.name());

        let cpu = CPU::<Base>::new();
        let cpu_buf = Buffer::from((&cpu, [1, 2, 4, 5]));

        let out = Buffer::from((&device, cpu_buf));
        assert_eq!(device.read(&out), [1, 2, 4, 5]);

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_buf_device_conversion_cu() -> crate::Result<()> {
        use crate::{Base, Buffer, Read, CPU, CUDA};

        let device = CUDA::<Base>::new(0)?;

        let cpu = CPU::<Base>::new();
        let cpu_buf = Buffer::from((&cpu, [1, 2, 4, 5]));

        let out = Buffer::from((&device, cpu_buf));
        assert_eq!(device.read(&out), [1, 2, 4, 5]);

        Ok(())
    }
}
