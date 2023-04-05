use crate::{
    flag::AllocFlag, Buffer, Device, GlobalCount, Graph, Ident, KeeperAble, KeeperReturn, PtrConv,
    PtrType, Shape,
};

#[derive(Default)]
pub struct Keeper<D: Device> {
    ptrs: Vec<Option<D::Ptr<u8, ()>>>,
}

impl<D: Device + PtrConv> Keeper<D> {
    pub fn get_existing_buf<'a, T, S: Shape>(
        &self,
        device: &'a D,
        ident: Ident,
    ) -> Buffer<'a, T, D, S> {
        let ptr = self
            .ptrs
            .get(ident.idx)
            .expect(&format!("Cannot get buffer with id {}", ident.idx))
            .as_ref()
            .expect("The requested buffer was deallocated before.");

        let ptr = unsafe { D::convert(ptr, AllocFlag::Wrapper) };

        Buffer {
            ptr,
            device: Some(device),
            ident,
        }
    }

    // keeper
    pub fn add_to_cache<T, S: Shape>(
        &mut self,
        graph: &mut Graph<GlobalCount>,
        ptr: &<D as Device>::Ptr<T, S>,
    ) -> Ident {
        graph.add_leaf(ptr.size());

        let ident = Ident {
            idx: self.ptrs.len(),
            len: ptr.size(),
        };

        let raw_ptr = Some(unsafe { D::convert(ptr, AllocFlag::Wrapper) });
        self.ptrs.push(raw_ptr);
        ident
    }

    #[inline]
    pub fn remove(&mut self, id: Ident) {
        *self
            .ptrs
            .get_mut(id.idx)
            .expect(&format!("Cannot get buffer with id {}", id.idx)) = None;
    }
}

impl<D: Device + KeeperReturn + PtrConv> KeeperAble<D> for Keeper<D> {
    #[inline]
    unsafe fn get_existing_buf<T, S: Shape>(device: &D, ident: Ident) -> Option<Buffer<T, D, S>> {
        Some(device.keeper().get_existing_buf(device, ident))
    }

    #[inline]
    fn remove(device: &D, id: Ident) {
        device.keeper_mut().remove(id)
    }

    #[inline]
    fn add_to_cache<T, S: Shape>(device: &D, ptr: &<D as Device>::Ptr<T, S>) -> Ident {
        device
            .keeper_mut()
            .add_to_cache(&mut device.graph_mut(), ptr)
    }
}
