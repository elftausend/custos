use crate::{
    cpu::CPUPtr, Addons, AddonsReturn, Alloc, Buffer, Cache, CacheReturn, Device, Ident, PtrConv,
    Shape, CPU,
};

use super::NnapiPtr;
use core::cell::{Cell, RefCell};
use nnapi::{AsOperandCode, Compilation, Execution, Model, Operand};

type ArrayId = (u32, ArrayPtr);

/// Used to run operations performed by the NNAPI.
/// It represents a single model.
pub struct NnapiDevice {
    /// The NNAPI model.
    pub model: RefCell<Model>,
    operand_count: Cell<u32>,
    /// An array of pointers with a corresponding index in the NNAPI model.
    pub input_ptrs: RefCell<Vec<ArrayId>>,
    compilation: RefCell<Option<Compilation>>,
    addons: Addons<Self>,
}

impl Device for NnapiDevice {
    type Ptr<U, S: crate::Shape> = NnapiPtr;

    type Cache = Cache<NnapiDevice>;

    #[inline]
    fn new() -> crate::Result<Self> {
        NnapiDevice::new()
    }
}

/// A [`CPUPtr`] with a u8 generic type.
pub type ArrayPtr = CPUPtr<u8>;

/// Creates an [`Operand`] (datatype) from a shape `S`.
#[inline]
pub fn dtype_from_shape<'a, T: AsOperandCode, S: Shape>() -> Operand {
    let dims = S::dims()
        .into_iter()
        .map(|dim| dim as u32)
        .collect::<Vec<u32>>();
    Operand::tensor(T::OPERAND_CODE, dims, 0., 0)
}

impl<'a, T: AsOperandCode, S: Shape> Alloc<'a, T, S> for NnapiDevice {
    fn alloc(&'a self, _len: usize, flag: crate::flag::AllocFlag) -> <Self as Device>::Ptr<T, S> {
        let dtype = dtype_from_shape::<T, S>();
        let idx = self.add_operand(&dtype).unwrap();
        NnapiPtr { dtype, idx, flag }
    }

    fn with_slice(&'a self, data: &[T]) -> <Self as Device>::Ptr<T, S>
    where
        T: Clone,
    {
        let nnapi_ptr = Alloc::<T, S>::alloc(self, data.len(), crate::flag::AllocFlag::default());

        let mut ptr = unsafe { CPUPtr::<T>::new(data.len(), crate::flag::AllocFlag::Wrapper) };
        ptr.clone_from_slice(data);
        let ptr = unsafe { CPU::convert::<T, S, u8, S>(&ptr, crate::flag::AllocFlag::None) };

        self.input_ptrs.borrow_mut().push((nnapi_ptr.idx, ptr));
        nnapi_ptr
    }
}

impl NnapiDevice {
    /// Creates a new [`NnapiDevice`].
    pub fn new() -> crate::Result<Self> {
        Ok(Self {
            model: RefCell::new(Model::new()?),
            operand_count: Cell::new(0),
            addons: Addons::default(),
            input_ptrs: Default::default(),
            compilation: Default::default(),
        })
    }

    /// Compiles the model and stores it in the [`NnapiDevice`].
    /// It handles setting the inputs and outputs of the model.
    pub fn compile<T, S: Shape>(&self, out: Buffer<T, Self, S>) -> crate::Result<()> {
        let mut model = self.model.borrow_mut();

        let input_ids = self
            .input_ptrs
            .borrow()
            .iter()
            .map(|(id, _)| *id)
            .collect::<Vec<u32>>();

        // let NnapiPtr { dtype, idx } = self.last_output.borrow().unwrap();

        model.identify_inputs_and_outputs(&input_ids, &[out.ptr.idx])?;

        model.finish()?;
        let mut compilation = model.compile()?;
        compilation.finish()?;
        *self.compilation.borrow_mut() = Some(compilation);
        Ok(())
    }

    fn set_input_ptrs(&self, run: &mut Execution) -> crate::Result<()> {
        for (idx, (_id, input_ptr)) in self.input_ptrs.borrow().iter().enumerate() {
            unsafe {
                run.set_input_raw(
                    idx as i32,
                    input_ptr.ptr.cast(),
                    input_ptr
                        .size
                        .expect("`size` is set during with_slice creation")
                        * input_ptr.len,
                )
            }?
        }
        Ok(())
    }

    /// Runs the model and returns the output.
    /// It reuses the same [`Compilation`] if it exists.
    pub fn run<T, S>(&self, out: Buffer<T, Self, S>) -> crate::Result<Vec<T>>
    where
        T: Default + Copy + AsOperandCode,
        S: Shape,
    {
        if self.compilation.borrow().is_none() {
            self.compile(out)?;
        }
        let mut compilation = self.compilation.borrow_mut();
        let compilation = compilation
            .as_mut()
            .expect("Should be set during compilation");

        let mut run = compilation.create_execution()?;
        self.set_input_ptrs(&mut run)?;

        let mut out = vec![T::default(); S::LEN];

        run.set_output(0, &mut out)?;

        run.compute()?;

        Ok(out)
    }

    /// Adds an operand to the model.
    pub fn add_operand(&self, dtype: &Operand) -> crate::Result<u32> {
        self.model.borrow_mut().add_operand(&dtype)?;
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

#[cfg(test)]
mod tests {
    use nnapi::{nnapi_sys::OperationCode, Operand};

    use crate::{bump_count, Buffer, CacheReturn, Device, Dim1, Ident, NnapiDevice, WithShape};

    #[test]
    fn test_running_nnapi_ops() -> crate::Result<()> {
        let device = NnapiDevice::new()?;

        let lhs = Buffer::with(&device, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let rhs = Buffer::with(&device, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        let mut out = Buffer::<f32, _, Dim1<10>>::new(&device, 0);

        let mut model = device.model.borrow_mut();

        let activation_idx = device.add_operand(&Operand::activation()).unwrap();

        model
            .set_activation_operand_value(activation_idx as i32)
            .unwrap();
        model.add_operation(
            OperationCode::ANEURALNETWORKS_ADD,
            &[lhs.ptr.idx, rhs.ptr.idx, activation_idx],
            &[out.ptr.idx],
        )?;

        let mut out2 = Buffer::<f32, _, Dim1<10>>::new(&device, 0);
        let activation_idx = device.add_operand(&Operand::activation()).unwrap();
        model
            .set_activation_operand_value(activation_idx as i32)
            .unwrap();
        model.add_operation(
            OperationCode::ANEURALNETWORKS_MUL,
            &[lhs.ptr.idx, out.ptr.idx, activation_idx],
            &[out2.ptr.idx],
        )?;

        device.run(out2)?;

        Ok(())
    }

    #[test]
    fn test_nnapi_device() -> crate::Result<()> {
        let device = super::NnapiDevice::new()?;

        let lhs = Buffer::with(&device, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let rhs = Buffer::with(&device, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        // a single operation
        /*let out = {
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
        };*/

        let out =
            device
                .cache_mut()
                .get::<i32, Dim1<10>>(&device, Ident::new(lhs.len()), (), bump_count);

        let activation_idx = device.add_operand(&Operand::activation()).unwrap();
        let mut model = device.model.borrow_mut();

        model
            .set_activation_operand_value(activation_idx as i32)
            .unwrap();
        model
            .add_operation(
                OperationCode::ANEURALNETWORKS_ADD,
                &[lhs.ptr.idx, rhs.ptr.idx, activation_idx],
                &[out.ptr.idx],
            )
            .unwrap();
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
