use crate::{Alloc, GraphReturn, Buffer};

impl<'a, T, D, const N: usize> From<(&'a D, [T; N])> for Buffer<'a, T, D>
where
    T: Clone,
    D: Alloc<T> + GraphReturn,
    // TODO: remove later for stack impl
    D::Ptr<T, 0>: Default,
{
    fn from(device_slice: (&'a D, [T; N])) -> Self {
        let len = device_slice.1.len();
        Buffer {
            // TODO: with_array()
            ptr: device_slice.0.with_slice(&device_slice.1),
            len,
            device: Some(device_slice.0),
            node: device_slice.0.graph().add_leaf(len),
            ..Default::default()
        }
    }
}

impl<'a, T, D, const N: usize> From<(&'a D, &[T; N])> for Buffer<'a, T, D>
where
    T: Clone,
    D: Alloc<T> + GraphReturn,
    D::Ptr<T, 0>: Default,
{
    fn from(device_slice: (&'a D, &[T; N])) -> Self {
        let len = device_slice.1.len();
        Buffer {
            // TODO: with_array()
            ptr: device_slice.0.with_slice(device_slice.1),
            len,
            device: Some(device_slice.0),
            node: device_slice.0.graph().add_leaf(len),
            ..Default::default()
        }
    }
}

impl<'a, T, D> From<(&'a D, &[T])> for Buffer<'a, T, D>
where
    T: Clone,
    D: Alloc<T> + GraphReturn,
    D::Ptr<T, 0>: Default,
{
    fn from(device_slice: (&'a D, &[T])) -> Self {
        let len = device_slice.1.len();
        Buffer {
            ptr: device_slice.0.with_slice(device_slice.1),
            len,
            device: Some(device_slice.0),
            node: device_slice.0.graph().add_leaf(len),
            ..Default::default()
        }
    }
}

impl<'a, T, D> From<(&'a D, Vec<T>)> for Buffer<'a, T, D>
where
    T: Clone,
    D: Alloc<T> + GraphReturn,
    D::Ptr<T, 0>: Default,
{
    fn from(device_vec: (&'a D, Vec<T>)) -> Self {
        let len = device_vec.1.len();
        Buffer {
            ptr: device_vec.0.alloc_with_vec(device_vec.1),
            len,
            device: Some(device_vec.0),
            node: device_vec.0.graph().add_leaf(len),
            ..Default::default()
        }
    }
}

impl<'a, T, D> From<(&'a D, &Vec<T>)> for Buffer<'a, T, D>
where
    T: Clone,
    D: Alloc<T> + GraphReturn,
    D::Ptr<T, 0>: Default,
{
    fn from(device_slice: (&'a D, &Vec<T>)) -> Self {
        let len = device_slice.1.len();
        Buffer {
            ptr: device_slice.0.with_slice(device_slice.1),
            len,
            device: Some(device_slice.0),
            node: device_slice.0.graph().add_leaf(len),
            ..Default::default()
        }
    }
}

/*
// TODO: check if Wrapper flag fits
// TODO: unsafe from raw parts fn?
impl<'a, T: Copy> From<(*mut T, usize)> for Buffer<'a, T> {
    fn from(info: (*mut T, usize)) -> Self {
        Buffer {
            ptr: (info.0, null_mut(), 0),
            len: info.1,
            // TODO: use static CPU device or add CPU device to tuple
            device: Device::default(),
            flag: BufFlag::Wrapper,
            p: PhantomData,
        }
    }
}

// TODO: unsafe?
/// A slice is wrapped into a buffer, hence buffer operations can be executed.
/// During these operations, the wrapped slice is updated. (which violates the safety rules / borrow checker of rust)
impl<'a, T> From<&mut [T]> for Buffer<'a, T> {
    fn from(slice: &mut [T]) -> Self {
        Buffer {
            ptr: (slice.as_mut_ptr(), null_mut(), 0),
            len: slice.len(),
            // TODO: use static CPU device or add CPU device to tuple
            device: Device::default(),
            flag: BufFlag::Wrapper,
            p: PhantomData,
        }
    }
}

// TODO: unsafe?
impl<'a, T, const N: usize> From<&mut [T; N]> for Buffer<'a, T> {
    fn from(slice: &mut [T; N]) -> Self {
        Buffer {
            ptr: (slice.as_mut_ptr(), null_mut(), 0),
            len: slice.len(),
            // TODO: use static CPU device or add CPU device to tuple
            device: Device::default(),
            flag: BufFlag::Wrapper,
            p: PhantomData,
        }
    }
}

// TODO: check if Wrapper flag fits
impl<'a, T: CDatatype> From<(*mut c_void, usize)> for Buffer<'a, T> {
    fn from(info: (*mut c_void, usize)) -> Self {
        Buffer {
            ptr: (null_mut(), info.0, 0),
            len: info.1,
            // TODO: use static CPU device or add CPU device to tuple
            device: Device::default(),
            flag: BufFlag::Wrapper,
            p: PhantomData,
        }
    }
}

// TODO: check if Wrapper flag fits
impl<'a, T: CDatatype> From<(u64, usize)> for Buffer<'a, T> {
    fn from(info: (u64, usize)) -> Self {
        Buffer {
            ptr: (null_mut(), null_mut(), info.0),
            len: info.1,
            // TODO: use static CPU device or add CPU device to tuple
            device: Device::default(),
            flag: BufFlag::Wrapper,
            p: PhantomData,
        }
    }
}
*/
