use crate::{shape::Shape, Alloc, BufFlag, Buffer};

impl<'a, T, D, const N: usize> From<(&'a D, [T; N])> for Buffer<'a, T, D>
where
    T: Clone,
    D: Alloc<'a, T>,
{
    fn from(device_slice: (&'a D, [T; N])) -> Self {
        let len = device_slice.1.len();
        Buffer {
            // TODO: with_array()
            ptr: device_slice.0.with_slice(&device_slice.1),
            len,
            device: Some(device_slice.0),
            //node: device_slice.0.graph().add_leaf(len),
            node: Default::default(),
            flag: BufFlag::default(),
        }
    }
}

impl<'a, T, D, const N: usize> From<(&'a D, &[T; N])> for Buffer<'a, T, D>
where
    T: Clone,
    D: Alloc<'a, T>,
{
    fn from(device_slice: (&'a D, &[T; N])) -> Self {
        let len = device_slice.1.len();
        Buffer {
            // TODO: with_array()
            ptr: device_slice.0.with_slice(device_slice.1),
            len,
            device: Some(device_slice.0),
            //node: device_slice.0.graph().add_leaf(len),
            node: Default::default(),
            flag: BufFlag::default(),
        }
    }
}

impl<'a, T, D, S: Shape> From<(&'a D, &[T])> for Buffer<'a, T, D, S>
where
    T: Clone,
    D: Alloc<'a, T, S>,
{
    fn from(device_slice: (&'a D, &[T])) -> Self {
        let len = device_slice.1.len();
        Buffer {
            ptr: device_slice.0.with_slice(device_slice.1),
            len,
            device: Some(device_slice.0),
            //node: device_slice.0.graph().add_leaf(len),
            node: Default::default(),
            flag: BufFlag::default(),
        }
    }
}

#[cfg(not(feature = "no-std"))]
impl<'a, T, D, S: Shape> From<(&'a D, Vec<T>)> for Buffer<'a, T, D, S>
where
    T: Clone,
    D: Alloc<'a, T, S>,
{
    fn from(device_vec: (&'a D, Vec<T>)) -> Self {
        let len = device_vec.1.len();
        Buffer {
            ptr: device_vec.0.alloc_with_vec(device_vec.1),
            len,
            device: Some(device_vec.0),
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
    D: Alloc<'a, T, S>
{
    fn from(device_slice: (&'a D, &Vec<T>)) -> Self {
        let len = device_slice.1.len();
        Buffer {
            ptr: device_slice.0.with_slice(device_slice.1),
            len,
            device: Some(device_slice.0),
            //node: device_slice.0.graph().add_leaf(len),
            node: Default::default(),
            flag: BufFlag::default(),
        }
    }
}
