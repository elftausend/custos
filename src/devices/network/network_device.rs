use crate::{
    Alloc, Buffer, ClearBuf, Device, DeviceError, Error, Graph, GraphReturn, Read, GlobalCount, Shape, PtrType, flag::AllocFlag,
};
use core::{
    cell::{RefCell, RefMut, Ref},
    marker::PhantomData,
};
use cuw::AsDataType;
use cuwanto_client as cuw;
use std::net::ToSocketAddrs;

pub struct Network<'a> {
    pub cuw_client: RefCell<cuw::Client>,
    pub graph: RefCell<Graph<GlobalCount>>,
    _p: PhantomData<&'a u8>,
}

impl<'a> Network<'a> {
    pub fn new<A: ToSocketAddrs>(addr: A, device: cuw::DeviceType) -> crate::Result<Network<'a>> {
        let mut cuw_client = cuw::Client::connect(addr)?;

        // should receive error if this is missing
        cuw_client
            .create_device(device, 0)?;

        Ok(Network {
            cuw_client: RefCell::new(cuw_client),
            graph: RefCell::new(Graph::new()),
            _p: PhantomData,
        })
    }
}

impl<'a> Device for Network<'a> {
    type Ptr<U, S: Shape> = NetworkArray<'a, U>;
    type Cache = ();

    fn new() -> crate::Result<Self> {
        Err(DeviceError::MissingAddress.into())
    }
}

impl<'a, T: cuw::AsDataType, S: Shape> Alloc<'a, T, S> for Network<'a> {
    fn alloc(&'a self, len: usize, flag: AllocFlag) -> <Self as Device>::Ptr<T, S> {
        let id = self
            .cuw_client
            .borrow_mut()
            .alloc_buf::<T>(len as u32)
            .unwrap();

        NetworkArray {
            id,
            client: &self.cuw_client,
            _p: PhantomData,
        }
    }

    fn with_slice(&'a self, data: &[T]) -> <Self as Device>::Ptr<T, S>
    where
        T: Clone,
    {
        let array: NetworkArray<T> = Alloc::<T, ()>::alloc(self, data.len(), AllocFlag::None);
        self.cuw_client
            .borrow_mut()
            .write_buf(array.id, data)
            .unwrap();
        array
    }
}

impl<'a> GraphReturn for Network<'a> {
    #[inline]
    fn graph(&self) -> Ref<Graph<GlobalCount>> {
        self.graph.borrow()
    }

    #[inline]
    fn graph_mut(&self) -> RefMut<Graph<GlobalCount>> {
        self.graph.borrow_mut()
    }
}

impl<'b, T: AsDataType + Clone + Default> Read<T> for Network<'b> {
    type Read<'a> = Vec<T>
    where
        T: 'a,
        Network<'b>: 'a;

    #[inline]
    fn read<'a>(&self, buf: &'a Buffer<T, Network<'b>>) -> Self::Read<'a> {
        self.read_to_vec(buf)
    }

    #[inline]
    fn read_to_vec(&self, buf: &Buffer<T, Network>) -> Vec<T>
    where
        T: Default + Clone,
    {
        self.cuw_client.borrow_mut().read_buf(buf.ptr.id).unwrap()
    }
}

impl<'a, T> ClearBuf<T> for Network<'a> {
    fn clear(&self, buf: &mut Buffer<T, Network<'a>>) {
        todo!()
    }
}

pub struct NetworkArray<'a, T> {
    id: cuw::BufId,
    client: &'a RefCell<cuw::client::Client>,
    _p: PhantomData<T>,
}

impl<'a, T> PtrType for NetworkArray<'a, T> {
    #[inline]
    fn size(&self) -> usize {
        self.id.len as usize
    }

    #[inline]
    fn flag(&self) -> AllocFlag {
        AllocFlag::None
    }
}

/*impl<'a, T> Dealloc<T> for NetworkArray<'a, T> {
    unsafe fn dealloc(&mut self, _len: usize) {
        self.client.borrow_mut().dealloc_buf(self.id).unwrap();
    }
}*/
