use crate::{shape::Shape, Alloc, BufFlag, Buffer, GraphReturn};

impl<'a, T, D, const N: usize> From<(&'a D, [T; N])> for Buffer<'a, T, D>
where
    T: Clone,
    // FIXME: In this case, GraphReturn acts as an "IsDynamic" trait, as GraphReturn is not implemented for Stack
    D: Alloc<'a, T> + GraphReturn,
{
    fn from((device, array): (&'a D, [T; N])) -> Self {
        let len = array.len();
        Buffer {
            // TODO: with_array()
            ptr: device.with_slice(&array),
            len,
            device: Some(device),
            //node: device.graph().add_leaf(len),
            node: Default::default(),
            flag: BufFlag::default(),
        }
    }
}

impl<'a, T, D, const N: usize> From<(&'a D, &[T; N])> for Buffer<'a, T, D>
where
    T: Clone,
    D: Alloc<'a, T> + GraphReturn,
{
    fn from((device, array): (&'a D, &[T; N])) -> Self {
        let len = array.len();
        Buffer {
            // TODO: with_array()
            ptr: device.with_slice(array),
            len,
            device: Some(device),
            //node: device.graph().add_leaf(len),
            node: Default::default(),
            flag: BufFlag::default(),
        }
    }
}

impl<'a, T, D, S: Shape> From<(&'a D, &[T])> for Buffer<'a, T, D, S>
where
    T: Clone,
    D: Alloc<'a, T, S> + GraphReturn,
{
    fn from((device, slice): (&'a D, &[T])) -> Self {
        let len = slice.len();
        Buffer {
            ptr: device.with_slice(slice),
            len,
            device: Some(device),
            //node: device.graph().add_leaf(len),
            node: Default::default(),
            flag: BufFlag::default(),
        }
    }
}

#[cfg(not(feature = "no-std"))]
impl<'a, T, D, S: Shape> From<(&'a D, Vec<T>)> for Buffer<'a, T, D, S>
where
    T: Clone,
    D: Alloc<'a, T, S> + GraphReturn,
{
    fn from((device, vec): (&'a D, Vec<T>)) -> Self {
        let len = vec.len();
        Buffer {
            ptr: device.alloc_with_vec(vec),
            len,
            device: Some(device),
            //node: device_vec.0.graph().add_leaf(len),
            node: Default::default(),
            flag: BufFlag::default(),
        }
    }
}

#[cfg(not(feature = "no-std"))]
impl<'a, T, D, S: Shape> From<(&'a D, &Vec<T>)> for Buffer<'a, T, D, S>
where
    T: Clone,
    D: Alloc<'a, T, S> + GraphReturn,
{
    fn from((device, vec): (&'a D, &Vec<T>)) -> Self {
        let len = vec.len();
        Buffer {
            ptr: device.with_slice(vec),
            len,
            device: Some(device),
            //node: device.graph().add_leaf(len),
            node: Default::default(),
            flag: BufFlag::default(),
        }
    }
}
