use crate::{
    flag::AllocFlag, Buffer, Device, GlobalCount, Graph, Ident, KeeperReturn, PtrConv,
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
            .expect(&format!("Cannot remove buffer with id {}. This indicates a bug in custos", id.idx)) = None;
    }
}

pub trait KeeperAble<D: Device> {
    /// May return an existing buffer using the provided [`Ident`].
    /// This function panics if no buffer with the provided [`Ident`] exists.
    ///
    /// # Safety
    /// This function is unsafe because it is possible to return multiple `Buffer` with `Ident` that share the same memory.
    /// If this function is called twice with the same `Ident`, the returned `Buffer` will be the same.
    /// Even though the return `Buffer`s are owned, this does not lead to double-frees (see [`AllocFlag`]).
    unsafe fn get_existing_buf<T, S: Shape>(device: &D, id: Ident) -> Option<Buffer<T, D, S>>;

    /// Removes a `Buffer` with the provided [`Ident`] from the cache.
    /// This function is internally called when a `Buffer` with [`AllocFlag`] `None` is dropped.
    fn remove(device: &D, ident: Ident);

    /// Adds a pointer that was allocated by [`Alloc`] to the cache and returns a new corresponding [`Ident`].
    /// This function is internally called when a `Buffer` with [`AllocFlag`] `None` is created.
    // TODO: rename function
    fn add<T, S: Shape>(device: &D, ptr: &D::Ptr<T, S>) -> Ident;
}

impl<D: Device> KeeperAble<D> for () {
    #[inline]
    fn remove(_device: &D, _ident: Ident) {}

    #[inline]
    fn add<T, S: Shape>(_device: &D, ptr: &<D as Device>::Ptr<T, S>) -> Ident {
        // None return?
        todo!()
        // Ident::new(ptr.size())
    }

    #[inline]
    unsafe fn get_existing_buf<T, S: Shape>(_device: &D, _id: Ident) -> Option<Buffer<T, D, S>> {
        None
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
    fn add<T, S: Shape>(device: &D, ptr: &<D as Device>::Ptr<T, S>) -> Ident {
        device
            .keeper_mut()
            .add_to_cache(&mut device.graph_mut(), ptr)
    }
}
