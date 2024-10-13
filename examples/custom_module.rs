use custos::{
    Alloc, Base, CowMut, Device, HasId, Module, OnDropBuffer, PtrType, Retrieve, Setup, Shape,
    Unit, WrappedData, CPU,
};

pub struct CustomModule<Mods> {
    pub mods: Mods,
}

// bare minimum to implement a custom module

impl<'a, D: 'a, Mods: Module<'a, D>> Module<'a, D> for CustomModule<Mods> {
    type Module = CustomModule<Mods::Module>;

    fn new() -> Self::Module {
        CustomModule { mods: Mods::new() }
    }
}

impl<Mods, D> Setup<D> for CustomModule<Mods>
where
    Mods: Setup<D>,
{
    fn setup(device: &mut D) -> custos::Result<()> {
        Mods::setup(device)
    }
}

fn main() {
    let _dev = CPU::<CustomModule<Base>>::new();
    // this is everything you need to do to implement a custom module
    // for actual usage, implement pass down traits / features (Retrieve, AddOperation, .., custom ones, ..)
    // This is demonstrated below the main function.
}

// Implementing pass down traits / features

impl<Mods: WrappedData> WrappedData for CustomModule<Mods> {
    type Wrap<T, Base: HasId + PtrType> = Mods::Wrap<T, Base>;

    #[inline]
    fn wrap_in_base<T, Base: HasId + PtrType>(&self, base: Base) -> Self::Wrap<T, Base> {
        self.mods.wrap_in_base(base)
    }

    #[inline]
    fn wrapped_as_base<T, Base: HasId + PtrType>(wrap: &Self::Wrap<T, Base>) -> &Base {
        Mods::wrapped_as_base(wrap)
    }

    #[inline]
    fn wrapped_as_base_mut<T, Base: HasId + PtrType>(wrap: &mut Self::Wrap<T, Base>) -> &mut Base {
        Mods::wrapped_as_base_mut(wrap)
    }
}

impl<Mods: OnDropBuffer + WrappedData> OnDropBuffer for CustomModule<Mods> {
    #[inline]
    fn on_drop_buffer<T: Unit, D: Device, S: Shape>(
        &self,
        _device: &D,
        _buf: &custos::prelude::Buffer<T, D, S>,
    ) {
        self.mods.on_drop_buffer(_device, _buf)
    }
}

impl<Mods, D, T, S> Retrieve<D, T, S> for CustomModule<Mods>
where
    T: Unit,
    Mods: Retrieve<D, T, S>,
    S: Shape,
{
    #[inline]
    unsafe fn retrieve<const NUM_PARENTS: usize>(
        &self,
        device: &D,
        len: usize,
        parents: impl custos::Parents<NUM_PARENTS>,
    ) -> custos::Result<CowMut<Self::Wrap<T, <D>::Base<T, S>>>>
    where
        S: Shape,
        D: Device + Alloc<T>,
    {
        // inject custom behaviour in this body

        self.mods.retrieve(device, len, parents)
    }

    fn on_retrieve_finish(&self, _retrieved_buf: &custos::prelude::Buffer<T, D, S>)
    where
        D: Alloc<T>,
    {
        // inject custom behaviour in this body

        self.mods.on_retrieve_finish(_retrieved_buf)
    }
}
