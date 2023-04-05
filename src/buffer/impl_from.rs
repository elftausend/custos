use core::ops::Range;

use crate::{number::Number, shape::Shape, Alloc, Buffer};

#[cfg(feature = "cpu")]
use crate::{WriteBuf, CPU};

impl<'a, T, D, const N: usize> From<(&'a D, [T; N])> for Buffer<'a, T, D>
where
    T: Clone,
    // TODO: IsShapeIndep ... find way to include Stack
    D: Alloc<'a, T>,
{
    #[inline]
    fn from((device, array): (&'a D, [T; N])) -> Self {
        Buffer::from_slice(device, &array)
    }
}

impl<'a, T, D> From<(&'a D, usize)> for Buffer<'a, T, D>
where
    D: Alloc<'a, T>,
{
    #[inline]
    fn from((device, len): (&'a D, usize)) -> Self {
        Buffer::new(&device, len)
    }
}

/*impl<'a, T, D> Buffer<'a, T, D>
where
    T: Clone,
    D: Alloc<'a, T>
{
    #[inline]
    pub fn from_iter<I: IntoIterator<Item = T>>(device: &'a D, iter: I) -> Self {
        Buffer::from_vec(&device, iter.into_iter().collect())
    }
}*/

impl<'a, T, D> From<(&'a D, Range<usize>)> for Buffer<'a, T, D>
where
    T: Number,
    D: Alloc<'a, T>,
{
    #[inline]
    fn from((device, range): (&'a D, Range<usize>)) -> Self {
        Buffer::from_vec(&device, range.map(|x| T::from_usize(x)).collect())
    }
}

/*impl<'a, T, D, const N: usize> From<(&'a D, [T; N])> for Buffer<'a, T, D>
where
    T: Clone,
    D: Alloc<'a, T> + IsShapeIndep,
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
    D: Alloc<'a, T>,
{
    #[inline]
    fn from((device, array): (&'a D, &[T; N])) -> Self {
        Buffer::from_slice(device, array)
    }
}

/*impl<'a, T, D, const N: usize> From<(&'a D, &[T; N])> for Buffer<'a, T, D>
where
    T: Clone,
    D: Alloc<'a, T> + IsShapeIndep,
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
    D: Alloc<'a, T, S>,
{
    #[inline]
    fn from((device, slice): (&'a D, &[T])) -> Self {
        Buffer::from_slice(device, slice)
    }
}

/*impl<'a, T, D, S: Shape> From<(&'a D, &[T])> for Buffer<'a, T, D, S>
where
    T: Clone,
    D: Alloc<'a, T, S> + IsShapeIndep,
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
    D: Alloc<'a, T, S>,
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
    D: Alloc<'a, T, S>,
{
    #[inline]
    fn from((device, vec): (&'a D, &Vec<T>)) -> Self {
        Buffer::from_slice(device, vec)
    }
}

#[cfg(feature = "cpu")]
impl<'a, 'b, T, S, D> From<(&'a D, Buffer<'b, T, CPU, S>)> for Buffer<'a, T, D, S>
where
    S: Shape,
    D: WriteBuf<T, S> + for<'c> Alloc<'c, T, S>,
{
    fn from((device, buf): (&'a D, Buffer<'b, T, CPU, S>)) -> Self {
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
        use crate::{Buffer, Read, CPU};

        let device = CPU::new();

        let cpu = CPU::new();
        let cpu_buf = Buffer::from((&cpu, [1, 2, 4, 5]));

        let out = Buffer::from((&device, cpu_buf));
        assert_eq!(device.read(&out), [1, 2, 4, 5]);
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_buf_device_conversion_cl() -> crate::Result<()> {
        use crate::{Buffer, OpenCL, Read, CPU};

        let device = OpenCL::new(0)?;

        let cpu = CPU::new();
        let cpu_buf = Buffer::from((&cpu, [1, 2, 4, 5]));

        let out = Buffer::from((&device, cpu_buf));
        assert_eq!(device.read(&out), [1, 2, 4, 5]);

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_buf_device_conversion_cu() -> crate::Result<()> {
        use crate::{Buffer, Read, CPU, CUDA};

        let device = CUDA::new(0)?;

        let cpu = CPU::new();
        let cpu_buf = Buffer::from((&cpu, [1, 2, 4, 5]));

        let out = Buffer::from((&device, cpu_buf));
        assert_eq!(device.read(&out), [1, 2, 4, 5]);

        Ok(())
    }
}
