use crate::{
    Alloc, Buffer, ClearBuf, Dealloc, Device, DeviceError, Error, Graph, GraphReturn, Read,
};
use core::{
    cell::{RefCell, RefMut},
    marker::PhantomData,
};
use cuw::AsDataType;
use cuwanto_server as cuw;
use std::net::ToSocketAddrs;

pub struct Network<'a> {
    pub cuw_client: RefCell<cuw::client::Client>,
    pub graph: RefCell<Graph>,
    _p: PhantomData<&'a u8>,
}

impl<'a> Network<'a> {
    pub fn new<A: ToSocketAddrs>(addr: A, device: cuw::DeviceType) -> crate::Result<Network<'a>> {
        let mut cuw_client = cuw::client::Client::connect(addr).map_err(|error| Error { error })?;

        // should receive error if this is missing
        cuw_client
            .create_device(device, 0)
            .map_err(|error| Error { error })?;

        Ok(Network {
            cuw_client: RefCell::new(cuw_client),
            graph: RefCell::new(Graph::new()),
            _p: PhantomData,
        })
    }
}

impl<'a> Device for Network<'a> {
    type Ptr<U, const N: usize> = NetworkArray<'a, U>;
    type Cache<const N: usize> = ();

    fn new() -> crate::Result<Self> {
        Err(DeviceError::MissingAddress.into())
    }
}

impl<'a, T: cuw::AsDataType> Alloc<'a, T> for Network<'a> {
    fn alloc(&'a self, len: usize) -> <Self as Device>::Ptr<T, 0> {
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

    fn with_slice(&'a self, data: &[T]) -> <Self as Device>::Ptr<T, 0>
    where
        T: Clone,
    {
        let array: NetworkArray<T> = self.alloc(data.len());
        self.cuw_client
            .borrow_mut()
            .write_buf(array.id, data)
            .unwrap();
        array
    }
}

impl<'a> GraphReturn for Network<'a> {
    #[inline]
    fn graph(&self) -> RefMut<Graph> {
        self.graph.borrow_mut()
    }
}

impl<'b, T: AsDataType + Clone + Default> Read<T, Network<'b>> for Network<'b> {
    type Read<'a> = Vec<T>
    where
        T: 'a,
        Network<'b>: 'a;

    fn read<'a>(&self, buf: &'a Buffer<T, Network<'b>>) -> Self::Read<'a> {
        self.read_to_vec(buf)
    }

    fn read_to_vec(&self, buf: &Buffer<T, Network>) -> Vec<T>
    where
        T: Default + Clone,
    {
        self.cuw_client.borrow_mut().read_buf(buf.ptr.id).unwrap()
    }
}

impl<'a, T> ClearBuf<T, Network<'a>> for Network<'a> {
    fn clear(&self, buf: &mut Buffer<T, Network<'a>>) {
        todo!()
    }
}

pub struct NetworkArray<'a, T> {
    id: cuw::client::BufId,
    client: &'a RefCell<cuw::client::Client>,
    _p: PhantomData<T>,
}

impl<'a, T> Dealloc<T> for NetworkArray<'a, T> {
    unsafe fn dealloc(&mut self, _len: usize) {
        self.client.borrow_mut().dealloc_buf(self.id).unwrap();
    }
}
