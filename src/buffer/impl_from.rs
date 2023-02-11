use crate::{shape::Shape, Alloc, Buffer, WriteBuf};

#[cfg(any(feature = "opencl", feature = "cuda"))]
use crate::{Device, CPU};

#[cfg(feature = "opencl")]
use crate::OpenCL;

#[cfg(feature = "cuda")]
use crate::CUDA;

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

#[cfg(feature = "opencl")]
impl<'a, 'b, T, S: Shape> From<(&'a OpenCL, Buffer<'b, T, CPU, S>)> for Buffer<'a, T, OpenCL> {
    fn from((device, buf): (&'a OpenCL, Buffer<'b, T, CPU, S>)) -> Self {
        let mut out = device.retrieve(buf.len());
        device.write(&mut out, &buf);
        out
    }
}

#[cfg(feature = "cuda")]
impl<'a, 'b, T, S: Shape> From<(&'a CUDA, Buffer<'b, T, CPU, S>)> for Buffer<'a, T, CUDA> {
    fn from((device, buf): (&'a CUDA, Buffer<'b, T, CPU, S>)) -> Self {
        let mut out = device.retrieve(buf.len());
        device.write(&mut out, &buf);
        out
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature="opencl")]
    #[test]
    fn test_buf_device_conversion_cl() -> crate::Result<()> {
        use crate::{OpenCL, Buffer, CPU, Read};

        let device = OpenCL::new(0)?;

        let cpu = CPU::new();
        let cpu_buf = Buffer::from((&cpu, [1, 2, 4, 5]));

        let out = Buffer::from((&device, cpu_buf));
        assert_eq!(device.read(&out), [1, 2, 4, 5]);

        Ok(())
    }

    #[cfg(feature="cuda")]
    #[test]
    fn test_buf_device_conversion_cu() -> crate::Result<()> {
        use crate::{CUDA, Buffer, CPU, Read};

        let device = CUDA::new(0)?;

        let cpu = CPU::new();
        let cpu_buf = Buffer::from((&cpu, [1, 2, 4, 5]));

        let out = Buffer::from((&device, cpu_buf));
        assert_eq!(device.read(&out), [1, 2, 4, 5]);

        Ok(())
    }
}
