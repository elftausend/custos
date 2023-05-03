use crate::{Addons, AddonsReturn, Alloc, Cache, Device, PtrConv, PtrType, Shape, Buffer};
use core::{
    cell::{Cell, RefCell},
    ffi::c_void,
    mem::{ManuallyDrop, size_of},
};
use nnapi::{AsOperandCode, Model, Operand, Compilation};


type ArrayId = (u32, ArrayPtr);

pub struct NnapiDevice {
    pub model: RefCell<Model>,
    operand_count: Cell<u32>,
    pub input_ptrs: RefCell<Vec<ArrayId>>,
    // last_output: RefCell<Option<NnapiPtr>>,
    compilation: RefCell<Option<Compilation>>,
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
            compilation: Default::default(),
            // last_output: Default::default(),
        })
    }

    pub fn compile<T, S: Shape>(&self, out: Buffer<T, Self, S>) -> crate::Result<()> {
        let mut model = self.model.borrow_mut();

        let input_ids = self.input_ptrs.borrow().iter().map(|(id, _)| *id).collect::<Vec<u32>>();

        // let NnapiPtr { dtype, idx } = self.last_output.borrow().unwrap();

        model.identify_inputs_and_outputs(&input_ids, &[out.ptr.idx])?;

        model.finish()?;
        let mut compilation = model.compile()?;
        compilation.finish()?;
        *self.compilation.borrow_mut() = Some(compilation);
        Ok(())
    }

    pub fn run<T, S>(&self, out: Buffer<T, Self, S>) -> crate::Result<Vec<T>> where T: Default + Copy + AsOperandCode, S: Shape {
        if self.compilation.borrow().is_none() {
            self.compile(out)?;
        }
        let mut compilation = self.compilation.borrow_mut();
        let compilation = compilation.as_mut().unwrap();

        let mut run = compilation.create_execution()?;

        for (idx, (_id, input_ptr)) in self.input_ptrs.borrow().iter().enumerate() {
            unsafe {
                run.set_input_raw(idx as i32, input_ptr.ptr, input_ptr.size * input_ptr.len)
            }?;
        }

        let mut out = vec![T::default(); S::LEN];

        unsafe {
            run.set_output_raw(0, out.as_mut_ptr().cast(), S::LEN * size_of::<T>())
        }?;

        run.compute()?;

        Ok(out)
    }

    pub fn add_operand(&self, dtype: Operand) -> crate::Result<u32> {
        self.model.borrow_mut().add_operand(dtype)?;
        let idx = self.operand_count.get();
        self.operand_count.set(idx + 1);
        Ok(idx)
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
    pub idx: u32,
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
    #[inline]
    fn size(&self) -> usize {
        self.dtype.len
    }

    #[inline]
    fn flag(&self) -> crate::flag::AllocFlag {
        crate::flag::AllocFlag::None
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

#[inline]
pub fn dtype_from_shape<T: AsOperandCode, S: Shape>() -> Operand {
    Operand::tensor(
        T::OPERAND_CODE,
        &S::dims()
            .into_iter()
            .map(|dim| dim as u32)
            .collect::<Vec<u32>>(),
        0.,
        0,
    )
}

impl<'a, T: AsOperandCode, S: Shape> Alloc<'a, T, S> for NnapiDevice {
    fn alloc(&'a self, _len: usize, _flag: crate::flag::AllocFlag) -> <Self as Device>::Ptr<T, S> {
        let dtype = dtype_from_shape::<T, S>();
        let idx = self.add_operand(dtype).unwrap();
        NnapiPtr { dtype, idx }
    }

    fn with_slice(&'a self, data: &[T]) -> <Self as Device>::Ptr<T, S>
    where
        T: Clone,
    {
        let nnapi_ptr = Alloc::<T, S>::alloc(self, data.len(), crate::flag::AllocFlag::default());

        let mut data = ManuallyDrop::new(data.to_vec());
        self.input_ptrs.borrow_mut().push((nnapi_ptr.idx, ArrayPtr {
            ptr: data.as_mut_ptr() as *mut c_void,
            len: data.len(),
            size: std::mem::size_of::<T>(),
        }));
        nnapi_ptr
    }
}

#[cfg(test)]
mod tests {
    use nnapi::{nnapi_sys::OperationCode, Operand};

    use crate::{Buffer, Device, Dim1, WithShape};

    #[test]
    fn test_nnapi_device() -> crate::Result<()> {
        let mut device = super::NnapiDevice::new()?;

        let lhs = Buffer::with(&device, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let rhs = Buffer::with(&device, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        
        // a single operation
        let out = {
            let out = device.retrieve::<i32, Dim1<10>>(lhs.len(), ());
            let activation_idx = device.add_operand(Operand::activation())?;
            let mut model = device.model.borrow_mut();

            model.set_activation_operand_value(activation_idx as i32)?;
            model.add_operation(
                OperationCode::ANEURALNETWORKS_ADD,
                &[lhs.ptr.idx, rhs.ptr.idx, activation_idx],
                &[out.ptr.idx],
            )?;
            out
        };

        // another one
        /*let out = {
            let out1 = device.retrieve::<i32, Dim1<10>>(lhs.len(), ());
            let activation_idx = device.add_operand(Operand::activation())?;
            let mut model = device.model.borrow_mut();

            model.set_activation_operand_value(activation_idx as i32)?;
            model.add_operation(
                OperationCode::ANEURALNETWORKS_ADD,
                &[out.ptr.idx, rhs.ptr.idx, activation_idx],
                &[out1.ptr.idx],
            )?;
            out1
        };*/

        device.run(out)?;   

        // assert_eq!(out.to_vec(), vec![2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

        Ok(())
    }
}
