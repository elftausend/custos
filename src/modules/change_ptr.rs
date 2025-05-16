use core::{cell::RefCell, hash::BuildHasherDefault, marker::PhantomData};
use std::collections::HashSet;

use crate::{
    impl_remove_layer, pass_down_add_operation, pass_down_cursor, pass_down_exec_now_module,
    pass_down_grad_fn, pass_down_replace_buf_module, pass_down_use_gpu_or_cpu, AddLayer, Alloc,
    Buffer, Cursor, Device, HasId, HasModules, Module, NoHasher, OnNewBuffer,
    Parents, PtrType, Retrieve, RunModule, Setup, Shape, UniqueId, Unit, WrappedData,
};


#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ChangePtr<Mods, Ptr> {
    pub modules: Mods,
    pub contains_ids: RefCell<HashSet<UniqueId, BuildHasherDefault<NoHasher>>>,
    pub pd: PhantomData<Ptr>,
}

impl<Mods: WrappedData, Ptr> WrappedData for ChangePtr<Mods, Ptr> {
    type Wrap<T: Unit, Base: HasId + PtrType> = Ptr;

    #[inline]
    fn wrap_in_base<T: Unit, Base: HasId + PtrType>(&self, base: Base) -> Self::Wrap<T, Base> {
        self.modules.wrap_in_base(base)
    }

    #[inline]
    fn wrapped_as_base<T: Unit, Base: HasId + PtrType>(wrap: &Self::Wrap<T, Base>) -> &Base {
        Mods::wrapped_as_base(wrap)
    }

    #[inline]
    fn wrapped_as_base_mut<T: Unit, Base: HasId + PtrType>(
        wrap: &mut Self::Wrap<T, Base>,
    ) -> &mut Base {
        Mods::wrapped_as_base_mut(wrap)
    }
}

impl<'a, Mods: Module<'a, D>, D: Device + 'a> Module<'a, D> for ChangePtr<Mods> {
    type Module = ChangePtr<Mods::Module>;

    fn new() -> Self::Module {
        ChangePtr {
            modules: Mods::new(),
            graph_trans: Default::default(),
            contains_ids: Default::default(),
            pd: PhantomData,
        }
    }
}

impl<Mods: Setup<D>, D> Setup<D> for ChangePtr<Mods> {
    #[inline]
    fn setup(device: &mut D) -> crate::Result<()> {
        Mods::setup(device)
    }
}

impl<Mods: Optimize> Optimize for ChangePtr<Mods> {
    fn optimize_mem_graph<D: 'static>(
        &self,
        device: &D,
        graph_translator: Option<&crate::GraphTranslator>,
    ) -> crate::Result<()> {
        match graph_translator {
            Some(graph_translator) => self
                .modules
                .optimize_mem_graph(device, Some(graph_translator)),
            None => {
                let graph_translator = self.graph_trans.borrow();
                self.modules
                    .optimize_mem_graph(device, Some(&graph_translator))
            }
        }
    }

    #[inline]
    fn unary_fusing<D: crate::UnaryFusing + 'static>(
        &self,
        device: &D,
        graph_translator: Option<&crate::modules::GraphTranslator>,
    ) -> crate::Result<()> {
        match graph_translator {
            Some(graph_translator) => self.modules.unary_fusing(device, Some(graph_translator)),
            None => {
                let graph_translator = self.graph_trans.borrow();
                self.modules.unary_fusing(device, Some(&graph_translator))
            }
        }
    }
}

impl<'a, Mods: OnNewBuffer<'a, T, D, S>, T: Unit, D: Device, S: Shape> OnNewBuffer<'a, T, D, S>
    for ChangePtr<Mods>
{
    unsafe fn on_new_buffer(&self, _device: &'a D, new_buf: &crate::Buffer<'a, T, D, S>) {
        let mut graph_trans = self.graph_trans.borrow_mut();
        let next_idx = graph_trans.next_idx;

        graph_trans.buf_id_to_idx.insert(new_buf.id().id, next_idx);
        graph_trans.add_leaf(new_buf.len());

        self.modules.on_new_buffer(_device, new_buf)
    }
}

pass_down_add_operation!(ChangePtr);
pass_down_exec_now_module!(ChangePtr);
#[cfg(feature = "cached")]
crate::pass_down_unified_mem_chain!(ChangePtr);
pass_down_use_gpu_or_cpu!(ChangePtr);
pass_down_replace_buf_module!(ChangePtr);
pass_down_grad_fn!(ChangePtr);

impl_remove_layer!(ChangePtr);

impl<Mods: RunModule<D>, D> RunModule<D> for ChangePtr<Mods> {
    #[inline]
    fn run(&self, _device: &D) -> crate::Result<()> {
        self.modules.run(_device)
    }
}

impl<NewMods, SD> AddLayer<NewMods, SD> for ChangePtr<()> {
    type Wrapped = crate::ChangePtr<NewMods>;

    #[inline]
    fn wrap_layer(inner_mods: NewMods) -> Self::Wrapped {
        ChangePtr {
            modules: inner_mods,
            graph_trans: Default::default(),
            contains_ids: Default::default(),
            pd: PhantomData,
        }
    }
}

impl<T, Mods, D, S> Retrieve<D, T, S> for ChangePtr<Mods>
where
    T: Unit + 'static,
    Mods: Retrieve<D, T, S>,
    D: Cursor + 'static,
    S: Shape,
{
    #[inline]
    unsafe fn retrieve<const NUM_PARENTS: usize>(
        &self,
        device: &D,
        len: usize,
        parents: impl Parents<NUM_PARENTS>,
    ) -> crate::Result<Self::Wrap<T, D::Base<T, S>>>
    where
        D: Alloc<T>,
    {
        let ids = parents.ids();
        let data = self.modules.retrieve(device, len, parents)?;

        let mut contains_ids = self.contains_ids.borrow_mut();

        // subtracting 1 because retrieving increments the cursor (cached and lazy modules)
        let cursor = device.cursor() as UniqueId - 1;

        if contains_ids.get(&cursor).is_some() {
            return Ok(data);
        }
        contains_ids.insert(cursor);

        let mut graph_trans = self.graph_trans.borrow_mut();

        let next_idx = graph_trans.next_idx;
        graph_trans.buf_id_to_idx.insert(data.id().id, next_idx);
        graph_trans.idx_to_buf_id.insert(next_idx, data.id().id);

        graph_trans.idx_to_cursor.insert(next_idx, cursor);

        // does a hash location check internally (again)
        graph_trans.add_node(len, &ids);
        Ok(data)
    }

    #[inline]
    fn on_retrieve_finish(&self, retrieved_buf: &Buffer<T, D, S>)
    where
        D: Alloc<T>,
    {
        // pass down
        self.modules.on_retrieve_finish(retrieved_buf)
    }
}

pass_down_cursor!(ChangePtr);

impl<Mods> HasModules for ChangePtr<Mods> {
    type Mods = Mods;

    #[inline]
    fn modules(&self) -> &Self::Mods {
        &self.modules
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "lazy")]
    #[cfg(feature = "cached")]
    #[test]
    fn test_lazy_graph_cached() {}
}
