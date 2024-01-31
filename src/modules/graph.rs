mod graph_translator;
mod node;
mod opt_graph;

pub use node::Node;
pub use opt_graph::*;

use core::{cell::RefCell, panic::Location};

use crate::{
    impl_remove_layer, pass_down_add_operation, pass_down_exec_now_module,
    pass_down_replace_buf_module, pass_down_use_gpu_or_cpu, AddLayer,
    Alloc, Buffer, Device, HasId, Module, OnDropBuffer, OnNewBuffer, OptimizeMemGraph, Parents,
    PtrType, Retrieve, RunModule, Setup, Shape, WrappedData,
};

pub use self::graph_translator::GraphTranslator;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Graph<Mods> {
    pub modules: Mods,
    pub graph_trans: RefCell<GraphTranslator>,
}

impl<Mods: WrappedData> WrappedData for Graph<Mods> {
    type Wrap<T, Base: HasId + PtrType> = Mods::Wrap<T, Base>;

    #[inline]
    fn wrap_in_base<T, Base: HasId + PtrType>(&self, base: Base) -> Self::Wrap<T, Base> {
        self.modules.wrap_in_base(base)
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

impl<Mods: Module<D>, D: Device> Module<D> for Graph<Mods> {
    type Module = Graph<Mods::Module>;

    fn new() -> Self::Module {
        Graph {
            modules: Mods::new(),
            graph_trans: Default::default(),
        }
    }
}

impl<Mods, D> Setup<D> for Graph<Mods> {
    fn setup(_device: &mut D) -> crate::Result<()> {
        Ok(())
    }
}

impl<Mods: OptimizeMemGraph> OptimizeMemGraph for Graph<Mods> {
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
}

impl<Mods: OnNewBuffer<T, D, S>, T, D: Device, S: Shape> OnNewBuffer<T, D, S> for Graph<Mods> {
    fn on_new_buffer(&self, _device: &D, new_buf: &crate::Buffer<T, D, S>) {
        let mut graph_trans = self.graph_trans.borrow_mut();
        let next_idx = graph_trans.next_idx;

        graph_trans.buf_id_to_idx.insert(new_buf.id().id, next_idx);
        graph_trans.add_leaf(new_buf.len());

        self.modules.on_new_buffer(_device, new_buf)
    }
}

impl<Mods: OnDropBuffer> OnDropBuffer for Graph<Mods> {
    #[inline]
    fn on_drop_buffer<T, D: Device, S: Shape>(&self, device: &D, buf: &crate::Buffer<T, D, S>) {
        self.modules.on_drop_buffer(device, buf)
    }
}

pass_down_add_operation!(Graph);
pass_down_exec_now_module!(Graph);
#[cfg(feature = "cached")]
crate::pass_down_unified_mem_chain!(Graph);
pass_down_use_gpu_or_cpu!(Graph);
pass_down_replace_buf_module!(Graph);

impl_remove_layer!(Graph);

impl<Mods: RunModule<D>, D> RunModule<D> for Graph<Mods> {
    #[inline]
    fn run(&self, _device: &D) -> crate::Result<()> {
        self.modules.run(_device)
    }
}

impl<NewMods, SD> AddLayer<NewMods, SD> for Graph<()> {
    type Wrapped = crate::Graph<NewMods>;

    #[inline]
    fn wrap_layer(inner_mods: NewMods) -> Self::Wrapped {
        Graph {
            modules: inner_mods,
            graph_trans: Default::default(),
        }
    }
}

impl<T: 'static, Mods: Retrieve<D, T, S>, D: 'static, S: Shape> Retrieve<D, T, S> for Graph<Mods> {
    #[inline]
    unsafe fn retrieve<const NUM_PARENTS: usize>(
        &self,
        device: &D,
        len: usize,
        parents: impl Parents<NUM_PARENTS>,
    ) -> Self::Wrap<T, D::Base<T, S>>
    where
        D: Alloc<T>,
    {
        let ids = parents.ids();
        let data = self.modules.retrieve(device, len, parents);
        let mut graph_trans = self.graph_trans.borrow_mut();

        if graph_trans
            .added_to_graph
            .contains(&Location::caller().into())
        {
            return data;
        }

        let next_idx = graph_trans.next_idx;
        graph_trans.buf_id_to_idx.insert(data.id().id, next_idx);
        graph_trans.idx_to_buf_id.insert(next_idx, data.id().id);

        graph_trans
            .idx_to_buf_location
            .insert(next_idx, Location::caller().into());

        // does a hash location check internally (again)
        graph_trans.add_node(len, &ids);
        data
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

#[cfg(test)]
mod tests {
    #[cfg(feature = "lazy")]
    #[cfg(feature = "cached")]
    #[test]
    fn test_lazy_graph_cached() {}
}
