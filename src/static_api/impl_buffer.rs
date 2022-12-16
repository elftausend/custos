use alloc::vec::Vec;

use crate::{Alloc, BufFlag, Buffer, GraphReturn, Node};

use super::static_cpu;

impl<'a, T: Clone> From<&[T]> for Buffer<'a, T> {
    fn from(slice: &[T]) -> Self {
        let device = static_cpu();
        Buffer {
            ptr: Alloc::<T>::with_slice(device, slice),
            len: slice.len(),
            device: Some(device),
            flag: BufFlag::None,
            node: device.graph().add_leaf(slice.len()),
        }
    }
}

impl<'a, T: Clone, const N: usize> From<&[T; N]> for Buffer<'a, T> {
    fn from(slice: &[T; N]) -> Self {
        let device = static_cpu();
        Buffer {
            //ptr: device.with_slice(slice),
            ptr: Alloc::<T>::with_slice(device, slice),
            len: slice.len(),
            device: Some(device),
            flag: BufFlag::None,
            node: device.graph().add_leaf(slice.len()),
        }
    }
}

impl<'a, T: Clone, const N: usize> From<[T; N]> for Buffer<'a, T> {
    fn from(slice: [T; N]) -> Self {
        let device = static_cpu();
        Buffer {
            //ptr: device.with_slice(&slice),
            ptr: Alloc::<T>::with_slice(device, &slice),
            len: slice.len(),
            device: Some(device),
            flag: BufFlag::None,
            node: Node::default(),
        }
    }
}

impl<'a, T: Clone> From<Vec<T>> for Buffer<'a, T> {
    fn from(data: Vec<T>) -> Self {
        let device = static_cpu();
        Buffer {
            len: data.len(),
            //ptr: device.alloc_with_vec(data),
            ptr: Alloc::<T>::alloc_with_vec(device, data),
            device: Some(device),
            flag: BufFlag::None,
            node: Node::default(),
        }
    }
}
