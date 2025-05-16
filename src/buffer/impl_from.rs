use core::ops::{Range, RangeInclusive};

use crate::{
    Alloc, Buffer, Device, OnNewBuffer, Retriever, Unit, WrappedData, number::Number, shape::Shape,
};

#[cfg(feature = "cpu")]
use crate::{CPU, WriteBuf};

impl<'a, T, D, const N: usize> From<(&'a D, [T; N])> for Buffer<'a, T, D>
where
    T: Unit + Clone,
    // TODO: IsShapeIndep ... find way to include Stack
    D: Alloc<T> + OnNewBuffer<'a, T, D, ()>,
{
    #[inline]
    fn from((device, array): (&'a D, [T; N])) -> Self {
        Buffer::from_slice(device, &array)
    }
}

impl<'a, T, D, S: Shape> From<(&'a D, usize)> for Buffer<'a, T, D, S>
where
    T: Unit,
    D: Alloc<T> + OnNewBuffer<'a, T, D, S>,
{
    #[inline]
    fn from((device, len): (&'a D, usize)) -> Self {
        Buffer::new(device, len)
    }
}

#[cfg(feature = "std")]
impl<'a, T, D, S> Buffer<'a, T, D, S>
where
    T: Unit + Clone,
    D: Alloc<T> + OnNewBuffer<'a, T, D, S>,
    S: Shape,
{
    #[inline]
    pub fn from_iter<I: IntoIterator<Item = T>>(device: &'a D, iter: I) -> Self {
        Buffer::from_vec(device, iter.into_iter().collect())
    }
}

#[cfg(feature = "std")]
impl<'a, T, D, S> From<(&'a D, Range<usize>)> for Buffer<'a, T, D, S>
where
    T: Number,
    D: Alloc<T> + OnNewBuffer<'a, T, D, S>,
    S: Shape,
{
    #[inline]
    fn from((device, range): (&'a D, Range<usize>)) -> Self {
        Buffer::from_iter(device, range.map(|x| T::from_usize(x)))
    }
}

#[cfg(feature = "std")]
impl<'a, T, D, S> From<(&'a D, RangeInclusive<usize>)> for Buffer<'a, T, D, S>
where
    T: Number,
    D: Alloc<T> + OnNewBuffer<'a, T, D, S>,
    S: Shape,
{
    #[inline]
    fn from((device, range): (&'a D, RangeInclusive<usize>)) -> Self {
        Buffer::from_iter(device, range.map(|x| T::from_usize(x)))
    }
}

impl<'a, T, D, const N: usize> From<(&'a D, &[T; N])> for Buffer<'a, T, D>
where
    T: Unit + Clone,
    // TODO: IsShapeIndep ... find way to include Stack
    D: Alloc<T> + OnNewBuffer<'a, T, D, ()>,
{
    #[inline]
    fn from((device, array): (&'a D, &[T; N])) -> Self {
        Buffer::from_slice(device, array)
    }
}

impl<'a, T, D, S: Shape> From<(&'a D, &[T])> for Buffer<'a, T, D, S>
where
    T: Unit + Clone,
    // TODO: IsShapeIndep ... find way to include Stack
    D: Alloc<T> + OnNewBuffer<'a, T, D, S>,
{
    #[inline]
    fn from((device, slice): (&'a D, &[T])) -> Self {
        Buffer::from_slice(device, slice)
    }
}

#[cfg(feature = "std")]
impl<'a, T, D, S: Shape> From<(&'a D, Vec<T>)> for Buffer<'a, T, D, S>
where
    T: Unit + Clone,
    // TODO: IsShapeIndep ... find way to include Stack
    D: Alloc<T> + OnNewBuffer<'a, T, D, S>,
{
    #[inline]
    fn from((device, vec): (&'a D, Vec<T>)) -> Self {
        Buffer::from_vec(device, vec)
    }
}

#[cfg(feature = "std")]
impl<'a, T, D, S: Shape> From<(&'a D, &Vec<T>)> for Buffer<'a, T, D, S>
where
    T: Unit + Clone,
    // TODO: IsShapeIndep ... find way to include Stack
    D: Alloc<T> + OnNewBuffer<'a, T, D, S>,
{
    #[inline]
    fn from((device, vec): (&'a D, &Vec<T>)) -> Self {
        Buffer::from_slice(device, vec)
    }
}

#[cfg(feature = "cpu")]
impl<'a, 'b, Mods: WrappedData, T, S, D> From<(&'a D, Buffer<'b, T, CPU<Mods>, S>)>
    for Buffer<'a, T, D, S>
where
    T: Unit + 'static,
    S: Shape,
    D: WriteBuf<T, S> + Device + Retriever<'a, T, S>,
{
    fn from((device, buf): (&'a D, Buffer<'b, T, CPU<Mods>, S>)) -> Self {
        let mut out = device.retrieve(buf.len(), &buf).unwrap();
        device.write(&mut out, buf.as_slice());
        out
    }
}

#[cfg(test)]
mod tests {

    #[cfg(feature = "cpu")]
    #[test]
    fn test_buf_device_conversion_cpu() {
        use crate::{Base, Buffer, CPU};

        let device = CPU::<Base>::new();

        let cpu = CPU::<Base>::new();
        let cpu_buf = Buffer::from((&cpu, [1, 2, 4, 5]));

        let out = Buffer::from((&device, cpu_buf));
        assert_eq!(out.read(), [1, 2, 4, 5]);
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_buf_device_conversion_cl() -> crate::Result<()> {
        use crate::{Base, Buffer, CPU, OpenCL, opencl::chosen_cl_idx};

        let device = OpenCL::<Base>::new(chosen_cl_idx())?;
        println!("name: {:?}", device.name());

        let cpu = CPU::<Base>::new();
        let cpu_buf = Buffer::from((&cpu, [1, 2, 4, 5]));

        let out = Buffer::from((&device, cpu_buf));
        assert_eq!(out.read(), [1, 2, 4, 5]);

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[cfg(feature = "cpu")]
    #[test]
    fn test_buf_device_conversion_cu() -> crate::Result<()> {
        use crate::{Base, Buffer, CPU, CUDA};

        let device = CUDA::<Base>::new(0)?;

        let cpu = CPU::<Base>::new();
        let cpu_buf = Buffer::from((&cpu, [1, 2, 4, 5]));

        let out = Buffer::from((&device, cpu_buf));
        assert_eq!(out.read(), [1, 2, 4, 5]);

        Ok(())
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_impl_from_range() {
        use crate::{CPU, Device};

        let device = CPU::based();
        let buffer = device.buffer::<f32, (), _>(4..10);
        assert_eq!(buffer.read(), [4., 5., 6., 7., 8., 9.,]);
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_impl_from_range_inclusive() {
        use crate::{CPU, Device};

        let device = CPU::based();
        let buffer = device.buffer::<f32, (), _>(4..=10);
        assert_eq!(buffer.read(), [4., 5., 6., 7., 8., 9., 10.,]);
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_impl_from_iter() {
        use crate::{Buffer, CPU};

        let device = CPU::based();
        let buffer = Buffer::<f32>::from_iter(
            &device,
            (0..=10)
                .into_iter()
                .filter(|x| x % 2 == 0)
                .map(|x| x as f32),
        );
        assert_eq!(buffer.read(), [0., 2., 4., 6., 8., 10.,]);
    }
}
