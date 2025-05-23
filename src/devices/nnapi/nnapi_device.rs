use crate::{
    Alloc, AsOperandCode, Base, Buffer, Device, IsShapeIndep, Lazy, LazyRun, LazySetup, Module,
    Retrieve, Retriever, Setup, Shape, Unit, WrappedData,
    cpu::{CPUPtr, DeallocWithLayout},
};

use super::NnapiPtr;
use core::cell::{Cell, RefCell};
use nnapi::{Compilation, Execution, Model, Operand};

type ArrayId = (u32, ArrayPtr);

/// Used to run operations performed by the NNAPI.
/// It represents a single model.
pub struct NnapiDevice<T, Mods = Base> {
    pub modules: Mods,
    /// The NNAPI model.
    pub model: RefCell<Model>,
    operand_count: Cell<u32>,
    /// An array of pointers with a corresponding index in the NNAPI model.
    pub input_ptrs: RefCell<Vec<ArrayId>>,
    pub last_created_ptr: RefCell<Option<NnapiPtr>>,
    compilation: RefCell<Option<Compilation>>,
    out: Cell<Vec<T>>,
}

impl<U, Mods: WrappedData> Device for NnapiDevice<U, Mods> {
    type Data<'a, T: Unit, S: crate::Shape> = Mods::Wrap<'a, T, NnapiPtr>;
    type Base<T: Unit, S: Shape> = NnapiPtr;
    type Error = crate::Error;

    #[inline]
    fn new() -> crate::Result<Self> {
        // NnapiDevice::new()
        todo!()
    }

    #[inline(always)]
    fn default_base_to_data<'a, T: Unit, S: Shape>(
        &'a self,
        base: Self::Base<T, S>,
    ) -> Self::Data<'a, T, S> {
        self.wrap_in_base(base)
    }

    #[inline(always)]
    fn default_base_to_data_unbound<'a, T: Unit, S: Shape>(
        &self,
        base: Self::Base<T, S>,
    ) -> Self::Data<'a, T, S> {
        self.wrap_in_base_unbound(base)
    }

    #[inline(always)]
    fn wrap_to_data<'a, T: Unit, S: Shape>(
        &self,
        wrap: Self::Wrap<'a, T, Self::Base<T, S>>,
    ) -> Self::Data<'a, T, S> {
        wrap
    }

    #[inline(always)]
    fn data_as_wrap<'a, 'b, T: Unit, S: Shape>(
        data: &'b Self::Data<'a, T, S>,
    ) -> &'b Self::Wrap<'a, T, Self::Base<T, S>> {
        data
    }

    #[inline(always)]
    fn data_as_wrap_mut<'a, 'b, T: Unit, S: Shape>(
        data: &'b mut Self::Data<'a, T, S>,
    ) -> &'b mut Self::Wrap<'a, T, Self::Base<T, S>> {
        data
    }
}

unsafe impl<U, Mods: WrappedData> IsShapeIndep for NnapiDevice<U, Mods> {}

impl<'a, U, T: Unit, D: Device, S: Shape, Mods: crate::OnNewBuffer<'a, T, D, S>>
    crate::OnNewBuffer<'a, T, D, S> for NnapiDevice<U, Mods>
{
    #[inline]
    fn on_new_buffer<'s>(&'a self, device: &'a D, new_buf: &'s mut Buffer<'a, T, D, S>) {
        self.modules.on_new_buffer(device, new_buf)
    }
}
impl<U, Mods: WrappedData> WrappedData for NnapiDevice<U, Mods> {
    type Wrap<'a, T: Unit, Base: crate::IsBasePtr> = Mods::Wrap<'a, T, Base>;

    #[inline]
    fn wrap_in_base<'a, T: Unit, Base: crate::IsBasePtr>(
        &'a self,
        base: Base,
    ) -> Self::Wrap<'a, T, Base> {
        self.modules.wrap_in_base(base)
    }

    #[inline]
    fn wrap_in_base_unbound<'a, T: Unit, Base: crate::IsBasePtr>(
        &self,
        base: Base,
    ) -> Self::Wrap<'a, T, Base> {
        self.modules.wrap_in_base_unbound(base)
    }

    #[inline]
    fn wrapped_as_base<'a, 'b, T: Unit, Base: crate::IsBasePtr>(
        wrap: &'b Self::Wrap<'a, T, Base>,
    ) -> &'b Base {
        Mods::wrapped_as_base(wrap)
    }

    #[inline]
    fn wrapped_as_base_mut<'a, 'b, T: Unit, Base: crate::IsBasePtr>(
        wrap: &'b mut Self::Wrap<'a, T, Base>,
    ) -> &'b mut Base {
        Mods::wrapped_as_base_mut(wrap)
    }
}

// impl<U, Mods: OnDropBuffer> OnDropBuffer for NnapiDevice<U, Mods> {
//     #[inline]
//     fn on_drop_buffer<T: Unit, D: Device, S: Shape>(&self, device: &D, buf: &Buffer<T, D, S>) {
//         self.modules.on_drop_buffer(device, buf)
//     }
// }

impl<U, Mods: Retrieve<Self, T, S>, T: AsOperandCode, S: Shape> Retriever<'_, T, S>
    for NnapiDevice<U, Mods>
{
    fn retrieve<const NUM_PARENTS: usize>(
        &self,
        len: usize,
        parents: impl crate::Parents<NUM_PARENTS>,
    ) -> crate::Result<Buffer<T, Self, S>> {
        let data = self.modules.retrieve::<NUM_PARENTS>(self, len, &parents)?;
        let buf = Buffer {
            data,
            device: Some(self),
        };
        self.modules.on_retrieve_finish(len, parents, &buf);
        Ok(buf)
    }
}

/// A [`CPUPtr`] with a u8 generic type.
pub type ArrayPtr = DeallocWithLayout;

/// Creates an [`Operand`] (datatype) from a shape `S`.
#[inline]
pub fn dtype_from_shape<T: AsOperandCode, S: Shape>() -> Operand {
    debug_assert!(S::LEN > 0);
    let dims = S::dims()
        .into_iter()
        .map(|dim| dim as u32)
        .collect::<Vec<u32>>();
    Operand::tensor(T::OPERAND_CODE, dims, 0., 0)
}

impl<U, T: AsOperandCode, Mods: WrappedData> Alloc<T> for NnapiDevice<U, Mods> {
    fn alloc<S: Shape>(
        &self,
        _len: usize,
        flag: crate::flag::AllocFlag,
    ) -> crate::Result<Self::Base<T, S>> {
        let dtype = dtype_from_shape::<T, S>();
        let idx = self.add_operand(&dtype)?;
        let nnapi_ptr = NnapiPtr { dtype, idx, flag };

        *self.last_created_ptr.borrow_mut() = Some(nnapi_ptr.clone());

        Ok(nnapi_ptr)
    }

    fn alloc_from_slice<S: Shape>(&self, data: &[T]) -> crate::Result<Self::Base<T, S>>
    where
        T: Clone,
    {
        let nnapi_ptr =
            Alloc::<T>::alloc::<S>(self, data.len(), crate::flag::AllocFlag::default())?;

        let mut ptr: CPUPtr<T> =
            unsafe { CPUPtr::<T>::new(data.len(), crate::flag::AllocFlag::Wrapper) };
        ptr.clone_from_slice(data);

        let ptr = unsafe { DeallocWithLayout::new(ptr) }.unwrap();

        self.input_ptrs.borrow_mut().push((nnapi_ptr.idx, ptr));
        Ok(nnapi_ptr)
    }
}

impl<T, SimpleMods> NnapiDevice<T, SimpleMods> {
    /// Creates a new [`NnapiDevice`].
    pub fn new<'a, NewMods>() -> crate::Result<NnapiDevice<T, NewMods>>
    // TODO keep in mind that lazy module requirement would make sense here
    where
        T: 'a,
        Self: 'a,
        SimpleMods: Module<'a, NnapiDevice<T>, Module = NewMods>,
        NewMods: Setup<NnapiDevice<T, NewMods>>,
    {
        let mut device = NnapiDevice {
            modules: SimpleMods::new(),
            model: RefCell::new(Model::new()?),
            operand_count: Cell::new(0),
            input_ptrs: Default::default(),
            last_created_ptr: RefCell::new(None),
            compilation: Default::default(),
            out: Default::default(),
        };

        Lazy::<NewMods>::setup(&mut device)?;

        Ok(device)
    }
}

impl<T, Mods: WrappedData> NnapiDevice<T, Mods> {
    /// Compiles the model and stores it in the [`NnapiDevice`].
    /// It handles setting the inputs and outputs of the model.
    pub fn compile(&self, out_idx: u32) -> crate::Result<()> {
        let mut model = self.model.borrow_mut();

        let input_ids = self
            .input_ptrs
            .borrow()
            .iter()
            .map(|(id, _)| *id)
            .collect::<Vec<u32>>();

        // let NnapiPtr { dtype, idx } = self.last_output.borrow().unwrap();

        model.identify_inputs_and_outputs(&input_ids, &[out_idx])?;

        model.finish()?;
        let mut compilation = model.compile()?;
        compilation.finish()?;
        *self.compilation.borrow_mut() = Some(compilation);
        Ok(())
    }

    fn set_input_ptrs(&self, run: &mut Execution) -> crate::Result<()> {
        for (idx, (_id, input_ptr)) in self.input_ptrs.borrow().iter().enumerate() {
            unsafe {
                run.set_input_raw(idx as i32, input_ptr.ptr.cast(), input_ptr.layout().size())
            }?
        }
        Ok(())
    }

    /// Runs the model and returns the output.
    /// It reuses the same [`Compilation`] if it exists.
    #[inline]
    pub fn run_with_vec(&self) -> crate::Result<Vec<T>>
    where
        T: Default + Copy + AsOperandCode,
    {
        LazyRun::run(self)?;
        Ok(self.out.take())
    }

    /// Adds an operand to the model.
    pub fn add_operand(&self, dtype: &Operand) -> crate::Result<u32> {
        self.model.borrow_mut().add_operand(dtype)?;
        let idx = self.operand_count.get();
        self.operand_count.set(idx + 1);
        Ok(idx)
    }
}

impl<T, Mods> LazySetup for NnapiDevice<T, Mods> {}

impl<'a, T: 'a> Default for NnapiDevice<T, Lazy<'a, Base>> {
    #[inline]
    fn default() -> Self {
        Self::new().unwrap()
    }
}

impl<T, Mods> LazyRun for NnapiDevice<T, Mods>
where
    T: Copy + Default + AsOperandCode,
    Mods: WrappedData,
{
    #[inline]
    fn run(&self) -> crate::Result<()> {
        if self.compilation.borrow().is_none() {
            self.compile(self.last_created_ptr.borrow().as_ref().unwrap().idx)?;
        }
        let mut compilation = self.compilation.borrow_mut();
        let compilation = compilation
            .as_mut()
            .expect("Should be set during compilation");

        let mut run = compilation.create_execution()?;
        self.set_input_ptrs(&mut run)?;

        // let mut out = vec![T::default(); S::LEN];
        let dtype = self
            .last_created_ptr
            .borrow()
            .as_ref()
            .unwrap()
            .dtype
            .clone();
        if T::OPERAND_CODE as i32 != dtype.inner.type_ {
            return Err("Datatype mismatch for output array".into());
        }

        let mut out = vec![T::default(); dtype.len];
        run.set_output(0, &mut out)?;
        self.out.set(out);

        run.compute()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use nnapi::{Operand, nnapi_sys::OperationCode};

    use crate::{Base, Buffer, Dim1, Lazy, LazyRun, NnapiDevice, WithShape};

    #[test]
    fn test_running_nnapi_ops() -> crate::Result<()> {
        let device = NnapiDevice::<i32, Lazy<Base>>::new()?;

        let lhs = Buffer::with(&device, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let rhs = Buffer::with(&device, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        let out = Buffer::<i32, _, Dim1<10>>::new(&device, 0);

        let mut model = device.model.borrow_mut();

        let activation_idx = device.add_operand(&Operand::activation()).unwrap();

        model
            .set_activation_operand_value(activation_idx as i32)
            .unwrap();
        model.add_operation(
            OperationCode::ANEURALNETWORKS_ADD,
            &[lhs.base().idx, rhs.base().idx, activation_idx],
            &[out.base().idx],
        )?;

        let out2 = Buffer::<f32, _, Dim1<10>>::new(&device, 0);
        let activation_idx = device.add_operand(&Operand::activation()).unwrap();
        model
            .set_activation_operand_value(activation_idx as i32)
            .unwrap();
        model.add_operation(
            OperationCode::ANEURALNETWORKS_MUL,
            &[lhs.base().idx, out.base().idx, activation_idx],
            &[out2.base().idx],
        )?;

        device.run()?;
        let out = device.out.take();

        assert_eq!(device.run_with_vec().unwrap(), out);

        Ok(())
    }
    /*
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
    }*/
}
