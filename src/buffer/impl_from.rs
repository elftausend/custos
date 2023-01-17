use crate::{shape::Shape, Alloc, Buffer, IsShapeIndep};

impl<'a, T, D, const N: usize> From<(&'a D, [T; N])> for Buffer<'a, T, D>
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
            node: Default::default(),
        }
    }
}

impl<'a, T, D, const N: usize> From<(&'a D, &[T; N])> for Buffer<'a, T, D>
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
            node: Default::default(),
        }
    }
}

impl<'a, T, D, S: Shape> From<(&'a D, &[T])> for Buffer<'a, T, D, S>
where
    T: Clone,
    D: Alloc<'a, T, S> + IsShapeIndep,
{
    fn from((device, slice): (&'a D, &[T])) -> Self {
        Buffer {
            ptr: device.with_slice(slice),
            device: Some(device),
            //node: device.graph().add_leaf(len),
            node: Default::default(),
        }
    }
}

#[cfg(not(feature = "no-std"))]
impl<'a, T, D, S: Shape> From<(&'a D, Vec<T>)> for Buffer<'a, T, D, S>
where
    T: Clone,
    D: Alloc<'a, T, S> + IsShapeIndep,
{
    fn from((device, vec): (&'a D, Vec<T>)) -> Self {
        Buffer {
            ptr: device.alloc_with_vec(vec),
            device: Some(device),
            //node: device_vec.0.graph().add_leaf(len),
            node: Default::default(),
        }
    }
}

#[cfg(not(feature = "no-std"))]
impl<'a, T, D, S: Shape> From<(&'a D, &Vec<T>)> for Buffer<'a, T, D, S>
where
    T: Clone,
    D: Alloc<'a, T, S> + IsShapeIndep,
{
    fn from((device, vec): (&'a D, &Vec<T>)) -> Self {
        Buffer {
            ptr: device.with_slice(vec),
            device: Some(device),
            //node: device.graph().add_leaf(len),
            node: Default::default(),
        }
    }
}
