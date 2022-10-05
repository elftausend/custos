use crate::{cpu::CPUPtr, Alloc, BufFlag, Buffer, GraphReturn, PtrType};

use super::static_cpu;

impl<'a, A> FromIterator<A> for Buffer<'a, A>
where
    A: Clone + Default,
{
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self {
        let device = static_cpu();
        let from_iter = Vec::from_iter(iter);

        Buffer {
            len: from_iter.len(),
            node: device.graph().add_leaf(from_iter.len()),
            ptr: CPUPtr::from_ptrs(device.alloc_with_vec(from_iter)),
            device: Some(device),
            flag: BufFlag::None,
        }
    }
}
