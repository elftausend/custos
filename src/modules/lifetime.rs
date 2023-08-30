mod tape;

use core::cell::RefCell;

use crate::{
    AddGradFn, ApplyFunction, Buffer, Device, Eval, HasId, MayToCLSource, Module, OnDropBuffer,
    Resolve, Retrieve, Retriever, Setup, Shape, TapeActions, CPU, OnNewBuffer,
};

use self::tape::Tape;

pub struct Lifetime<'a, Mods> {
    modules: Mods,
    tape: RefCell<Tape<'a>>,
}

impl<Mods: Module<D>, D> Module<D> for Lifetime<'_, Mods> {
    type Module<'a> = Lifetime<'a, Mods::Module<'a>>;

    fn new<'a>() -> Self::Module<'a> {
        todo!()
    }
}

impl<'a, Mods> OnDropBuffer for Lifetime<'a, Mods> {}
impl<'a, Mods, D> Setup<D> for Lifetime<'a, Mods> {}

impl<'a, Mods> AddGradFn<'a> for Lifetime<'a, Mods> {
    fn add_grad_fn<T, S: crate::Shape>(&self, grad_fn: impl Fn(&mut crate::Gradients) + 'a)
    where
        T: 'a,
    {
        self.tape.borrow_mut().add_grad_fn(grad_fn)
    }
}

impl<'a, Mods: AddGradFn<'a>> AddGradFn<'a> for CPU<Mods> {
    fn add_grad_fn<T, S: crate::Shape>(&'a self, grad_fn: impl Fn(&mut crate::Gradients) + 'a)
    where
        T: 'a,
    {
        self.modules.add_grad_fn::<T, S>(grad_fn)
    }
}

pub(crate) trait ApplyFunctionLifetimeTest<'a, T, S: Shape = (), D: Device = Self>:
    Device
{
    #[track_caller]
    fn apply_fn_lifetime<F>(
        &'a self,
        buf: &'a Buffer<T, D, S>,
        f: impl Fn(Resolve<T>) -> F, /*+ Copy*/
    ) -> Buffer<T, Self, S>
    where
        F: Eval<T> + MayToCLSource;
}

impl<'a, Mods, T, S> ApplyFunctionLifetimeTest<'a, T, S, Self> for CPU<Mods>
where
    CPU<Mods>: 'a,
    Mods: OnDropBuffer + AddGradFn<'a> + 'a,
    T: 'a,
    S: Shape + 'a,
{
    fn apply_fn_lifetime<F>(
        &'a self,
        buf: &'a Buffer<T, Self, S>,
        f: impl Fn(Resolve<T>) -> F,
    ) -> Buffer<'_, T, Self, S>
    where
        F: Eval<T> + MayToCLSource,
    {
        // let out = self.retrieve(buf.len(), buf);
        // let ids = (buf.id(), out.id());

        self.add_grad_fn::<T, S>(move |grads| {
            // grads.get_double::<T, S, S, Self>(ids);
            let x = buf;
        });

        todo!()
        // out
    }
}

impl<'a, T, D: Device, S: Shape, Mods: OnNewBuffer<T, D, S>> OnNewBuffer<T, D, S> for Lifetime<'a, Mods> {
    fn on_new_buffer(&self, device: &D, new_buf: &Buffer<T, D, S>) {
        self.modules.on_new_buffer(device, new_buf)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Base, Cached, CPU, Device};

    use super::{Lifetime, ApplyFunctionLifetimeTest};

    pub fn op() {
    }

    #[test]
    fn test_lifetime_module() {
        let cpu = CPU::<Lifetime<'_, Base>>::new();
        let buf: crate::Buffer<i32, CPU<Lifetime<'_, Base>>> = cpu.buffer::<i32, (), _>(&vec![1i32, 2]);
        cpu.apply_fn_lifetime(&buf, |x| x);

        drop(buf);
        drop(cpu);
    }
}
