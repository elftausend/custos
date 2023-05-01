use crate::{Addons, AddonsReturn, Alloc, Cache, Device, PtrConv, PtrType, Shape};
use core::cell::{RefCell, Cell};
use nnapi::{AsOperandCode, Model, Operand};

pub struct NnapiDevice {
    model: RefCell<Model>,
    operand_count: Cell<usize>,
    addons: Addons<Self>,
}

impl NnapiDevice {
    pub fn new() -> crate::Result<Self> {
        Ok(Self {
            model: RefCell::new(Model::new()?),
            operand_count: Cell::new(0),
            addons: Addons::default(),
        })
    }
}

impl Default for NnapiDevice {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

impl AddonsReturn for NnapiDevice {
    #[inline]
    fn addons(&self) -> &Addons<Self> {
        &self.addons
    }
}

pub struct NnapiPtr {
    dtype: Operand,
    idx: usize,
}

impl Default for NnapiPtr {
    fn default() -> Self {
        Self {
            dtype: Operand::activation(),
            idx: usize::MAX,
        }
    }
}

impl PtrConv for NnapiDevice {
    unsafe fn convert<T, IS: Shape, Conv, OS: Shape>(
        ptr: &Self::Ptr<T, IS>,
        flag: crate::flag::AllocFlag,
    ) -> Self::Ptr<Conv, OS> {
        NnapiPtr { dtype: ptr.dtype, idx: ptr.idx }
    }
}

impl PtrType for NnapiPtr {
    fn size(&self) -> usize {
        todo!()
    }

    fn flag(&self) -> crate::flag::AllocFlag {
        todo!()
    }
}

impl Device for NnapiDevice {
    type Ptr<U, S: crate::Shape> = NnapiPtr;

    type Cache = Cache<NnapiDevice>;

    fn new() -> crate::Result<Self> {
        todo!()
    }
}

impl<'a, T: AsOperandCode, S: Shape> Alloc<'a, T, S> for NnapiDevice {
    fn alloc(&'a self, len: usize, flag: crate::flag::AllocFlag) -> <Self as Device>::Ptr<T, S> {
        let dtype = Operand::tensor(
            T::OPERAND_CODE,
            &S::dims()
                .into_iter()
                .map(|dim| dim as u32)
                .collect::<Vec<u32>>(),
            0.,
            0,
        );
        self.model.borrow_mut().add_operand(dtype).unwrap();
        let idx = self.operand_count.get();
        self.operand_count.set(idx + 1);
        NnapiPtr { dtype, idx }
    }

    fn with_slice(&'a self, data: &[T]) -> <Self as Device>::Ptr<T, S>
    where
        T: Clone,
    {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::{Buffer, WithShape};

    #[test]
    fn test_nnapi_device() {
        let device = super::NnapiDevice::new().unwrap();

        let buf = Buffer::with(&device, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }
}
