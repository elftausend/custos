use crate::{Addons, AddonsReturn, Alloc, Cache, Device, PtrConv, PtrType, Shape};
use core::{
    cell::{Cell, RefCell},
    ffi::c_void,
    mem::ManuallyDrop,
};
use nnapi::{AsOperandCode, Model, Operand};

pub struct NnapiDevice {
    model: RefCell<Model>,
    operand_count: Cell<u32>,
    input_ptrs: RefCell<Vec<ArrayPtr>>,
    last_output: RefCell<Option<ArrayPtr>>,
    addons: Addons<Self>,
}

#[derive(Debug, Clone, Copy)]
pub struct ArrayPtr {
    ptr: *mut c_void,
    len: usize,
    size: usize,
}

impl NnapiDevice {
    pub fn new() -> crate::Result<Self> {
        Ok(Self {
            model: RefCell::new(Model::new()?),
            operand_count: Cell::new(0),
            addons: Addons::default(),
            input_ptrs: Default::default(),
            last_output: Default::default(),
        })
    }

    pub fn run(&self) -> crate::Result<()> {
        let mut model = self.model.borrow_mut();
        let mut compilation = model.compile()?;
        compilation.finish()?;
        let mut run = compilation.create_execution()?;

        for (idx, input_ptr) in self.input_ptrs.borrow().iter().enumerate() {
            unsafe {
                run.set_input_raw(idx as i32, input_ptr.ptr, input_ptr.size * input_ptr.len)
            }?;
        }

        let output_ptr = self.last_output.borrow().unwrap();
        unsafe {
            run.set_output_raw(0, output_ptr.ptr, output_ptr.size * output_ptr.len)
        }?;

        run.compute()?;

        Ok(())
    }
}

impl Default for NnapiDevice {
    #[inline]
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
    idx: u32,
}

impl Default for NnapiPtr {
    fn default() -> Self {
        Self {
            dtype: Operand::activation(),
            idx: u32::MAX,
        }
    }
}

impl PtrConv for NnapiDevice {
    unsafe fn convert<T, IS: Shape, Conv, OS: Shape>(
        ptr: &Self::Ptr<T, IS>,
        flag: crate::flag::AllocFlag,
    ) -> Self::Ptr<Conv, OS> {
        NnapiPtr {
            dtype: ptr.dtype,
            idx: ptr.idx,
        }
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

    #[inline]
    fn new() -> crate::Result<Self> {
        NnapiDevice::new()
    }
}

impl<'a, T: AsOperandCode, S: Shape> Alloc<'a, T, S> for NnapiDevice {
    fn alloc(&'a self, _len: usize, _flag: crate::flag::AllocFlag) -> <Self as Device>::Ptr<T, S> {
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
        let nnapi_ptr = Alloc::<T, S>::alloc(self, data.len(), crate::flag::AllocFlag::default());

        let mut data = ManuallyDrop::new(data.to_vec());
        self.input_ptrs.borrow_mut().push(ArrayPtr {
            ptr: data.as_mut_ptr() as *mut c_void,
            len: data.len(),
            size: std::mem::size_of::<T>(),
        });
        nnapi_ptr
    }
}

#[cfg(test)]
mod tests {
    use nnapi::nnapi_sys::OperationCode;

    use crate::{Buffer, Device, Dim1, WithShape};

    #[test]
    fn test_nnapi_device() -> crate::Result<()> {
        let device = super::NnapiDevice::new()?;

        let lhs = Buffer::with(&device, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let rhs = Buffer::with(&device, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        let mut out = device.retrieve::<i32, Dim1<10>>(lhs.len(), ());

        device.model.borrow_mut().add_operation(
            OperationCode::ANEURALNETWORKS_ADD,
            &[lhs.ptr.idx, rhs.ptr.idx],
            &[out.ptr.idx],
        )?;

        device.run()?;

        Ok(())
    }
}
