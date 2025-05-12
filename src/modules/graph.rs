mod graph_translator;
mod node;
mod opt_graph;

pub use node::Node;
pub use opt_graph::*;

use core::{cell::RefCell, hash::BuildHasherDefault, marker::PhantomData};
use std::collections::HashSet;

use crate::{
    AddLayer, Alloc, Buffer, Cursor, Device, HasId, HasModules, IsBasePtr, Module, NoHasher,
    OnNewBuffer, Optimize, Parents, ReplaceBufPassDown, Retrieve, RunModule, Setup, Shape,
    UniqueId, Unit, WrappedData, impl_remove_layer, pass_down_add_operation, pass_down_cursor,
    pass_down_grad_fn, pass_down_use_gpu_or_cpu,
};

pub use self::graph_translator::GraphTranslator;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Graph<Mods, T = f32> {
    pub modules: Mods,
    pub graph_trans: RefCell<GraphTranslator>,
    pub contains_ids: RefCell<HashSet<UniqueId, BuildHasherDefault<NoHasher>>>,
    pub pd: PhantomData<T>,
}

impl<Mods: WrappedData> WrappedData for Graph<Mods> {
    type Wrap<'a, T: Unit, Base: IsBasePtr> = Mods::Wrap<'a, T, Base>;

    #[inline]
    fn wrap_in_base<'a, T: Unit, Base: IsBasePtr>(&'a self, base: Base) -> Self::Wrap<'a, T, Base> {
        self.modules.wrap_in_base(base)
    }

    #[inline]
    fn wrap_in_base_unbound<'a, T: Unit, Base: IsBasePtr>(
        &self,
        base: Base,
    ) -> Self::Wrap<'a, T, Base> {
        self.modules.wrap_in_base_unbound(base)
    }

    #[inline]
    fn wrapped_as_base<'a, 'b, T: Unit, Base: IsBasePtr>(
        wrap: &'b Self::Wrap<'a, T, Base>,
    ) -> &'b Base {
        Mods::wrapped_as_base(wrap)
    }

    #[inline]
    fn wrapped_as_base_mut<'a, 'b, T: Unit, Base: IsBasePtr>(
        wrap: &'b mut Self::Wrap<'a, T, Base>,
    ) -> &'b mut Base {
        Mods::wrapped_as_base_mut(wrap)
    }
}

impl<'a, Mods: Module<'a, D>, D: Device + 'a> Module<'a, D> for Graph<Mods> {
    type Module = Graph<Mods::Module>;

    fn new() -> Self::Module {
        Graph {
            modules: Mods::new(),
            graph_trans: Default::default(),
            contains_ids: Default::default(),
            pd: PhantomData,
        }
    }
}

impl<Mods: Setup<D>, D> Setup<D> for Graph<Mods> {
    #[inline]
    fn setup(device: &mut D) -> crate::Result<()> {
        Mods::setup(device)
    }
}

impl<Mods: Optimize> Optimize for Graph<Mods> {
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
    for Graph<Mods>
{
    fn on_new_buffer(&'a self, _device: &'a D, new_buf: &mut crate::Buffer<'a, T, D, S>) {
        let mut graph_trans = self.graph_trans.borrow_mut();
        let next_idx = graph_trans.next_idx;

        graph_trans.buf_id_to_idx.insert(new_buf.id().id, next_idx);
        graph_trans.add_leaf(new_buf.len());

        self.modules.on_new_buffer(_device, new_buf)
    }
}

pass_down_add_operation!(Graph);
#[cfg(feature = "cached")]
crate::pass_down_unified_mem_chain!(Graph);
pass_down_use_gpu_or_cpu!(Graph);
pass_down_grad_fn!(Graph);

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
            contains_ids: Default::default(),
            pd: PhantomData,
        }
    }
}

impl<Mods: WrappedData> Graph<Mods> {
    pub fn retrieve_inner<'a, D, T, S, const NUM_PARENTS: usize>(
        &self,
        device: &D,
        len: usize,
        parents: &impl Parents<NUM_PARENTS>,
        retrieve_cb: impl Fn() -> crate::Result<Mods::Wrap<'a, T, D::Base<T, S>>>,
    ) -> crate::Result<<Self as WrappedData>::Wrap<'a, T, D::Base<T, S>>>
    where
        D: Device + Cursor,
        T: Unit,
        S: Shape,
    {
        let ids = parents.ids();
        let data = retrieve_cb()?;

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
}

impl<T, Mods, D, S> Retrieve<D, T, S> for Graph<Mods>
where
    T: Unit + 'static,
    Mods: Retrieve<D, T, S>,
    D: Cursor + 'static,
    S: Shape,
{
    #[inline]
    fn retrieve_entry<'a, const NUM_PARENTS: usize>(
        &'a self,
        device: &D,
        len: usize,
        parents: &impl Parents<NUM_PARENTS>,
    ) -> crate::Result<Self::Wrap<'a, T, D::Base<T, S>>>
    where
        D: Alloc<T>,
    {
        self.retrieve_inner(device, len, parents, || {
            self.modules.retrieve_entry(device, len, parents)
        })
    }

    #[inline]
    fn retrieve<'a, const NUM_PARENTS: usize>(
        &self,
        device: &D,
        len: usize,
        parents: &impl Parents<NUM_PARENTS>,
    ) -> crate::Result<Self::Wrap<'a, T, D::Base<T, S>>>
    where
        D: Alloc<T>,
    {
        self.retrieve_inner(device, len, parents, || {
            self.modules.retrieve(device, len, parents)
        })
    }

    #[inline]
    fn on_retrieve_finish<'a, const NUM_PARENTS: usize>(
        &self,
        len: usize,
        parents: impl Parents<NUM_PARENTS>,
        retrieved_buf: &Buffer<T, D, S>,
    ) where
        D: Alloc<T>,
    {
        self.modules.on_retrieve_finish(len, parents, retrieved_buf)
    }
}

pass_down_cursor!(Graph);
impl<Mods> ReplaceBufPassDown for Graph<Mods> {}

impl<Mods> HasModules for Graph<Mods> {
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
