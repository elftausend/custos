mod graph_translator;
mod node;
mod opt_graph;

use core::cell::RefCell;

use crate::{
    pass_down_add_operation, pass_down_exec_now_module, pass_down_unified_mem_chain,
    pass_down_use_gpu_or_cpu, Alloc, Buffer, Device, HasId, Module, OnDropBuffer, OnNewBuffer,
    Parents, PtrConv, Retrieve, Setup, Shape,
};

use self::graph_translator::GraphTranslator;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Graph<Mods> {
    modules: Mods,
    graph_trans: RefCell<GraphTranslator>,
}

impl<Mods: Module<D>, D> Module<D> for Graph<Mods> {
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
pass_down_unified_mem_chain!(Graph);
pass_down_use_gpu_or_cpu!(Graph);

impl<T: 'static, Mods: Retrieve<D, T>, D: PtrConv + 'static> Retrieve<D, T> for Graph<Mods> {
    #[inline]
    fn retrieve<S, const NUM_PARENTS: usize>(
        &self,
        device: &D,
        len: usize,
        parents: impl Parents<NUM_PARENTS>,
    ) -> <D>::Data<T, S>
    where
        S: Shape,
        D: Alloc<T>,
    {
        let data = self.modules.retrieve(device, len, parents);
        let mut graph_trans = self.graph_trans.borrow_mut();

        let next_idx = graph_trans.next_idx;
        graph_trans.buf_id_to_idx.insert(data.id().id, next_idx);

        graph_trans.add_node(len, &parents);
        data
    }

    #[inline]
    fn on_retrieve_finish<S: Shape>(&self, retrieved_buf: &Buffer<T, D, S>)
    where
        D: Alloc<T>,
    {
        // pass down
        self.modules.on_retrieve_finish(retrieved_buf)
    }
}

/*

use core::cell::{Ref, RefMut};

#[cfg(feature = "opt-cache")]
use crate::{CacheReturn, DeviceError};

// pub use add_graph::*;
pub use node::*;

mod node;

#[cfg(not(feature = "no-std"))]
mod graph_struct;

#[cfg(not(feature = "no-std"))]
pub use graph_struct::*;

/// Returns the next index for a [`Node`].
pub trait NodeIdx {
    /// Returns the next index for a [`Node`].
    #[inline]
    fn idx(nodes: &[Node]) -> usize {
        nodes.len()
    }
}

/// Uses the global count as the next index for a [`Node`].
#[derive(Debug, Default)]
pub struct GlobalCount;

#[cfg(feature = "no-std")]
impl NodeIdx for GlobalCount {}

/// A dummy graph for no-std.
#[cfg(feature = "no-std")]
pub struct Graph<IdxFrom: NodeIdx> {
    _p: core::marker::PhantomData<IdxFrom>,
}

#[cfg(feature = "no-std")]
impl<IdxFrom: NodeIdx> Graph<IdxFrom> {
    /// This function will panic. Disable the `no-std` feature to use this function.
    #[inline]
    pub fn add_leaf(&mut self, _len: usize) -> Node {
        unimplemented!("Not available in no-std mode")
    }

    /// This function will panic. Disable the `no-std` feature to use this function.
    #[inline]
    pub fn add_node(&mut self, _len: usize, _lhs_idx: usize, _rhs_idx: usize) -> Node {
        unimplemented!("Not available in no-std mode")
    }
}

/// A `CacheTrace` is a list of nodes that shows which [`Buffer`](crate::Buffer)s could use the same cache.
#[cfg(not(feature = "no-std"))]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CacheTrace {
    /// This identifier is the common cache index / ident. All the other idents in `use_cache_ids` can use this ident to share memory.
    pub cache_id: Ident,
    /// The identifiers of the nodes that can use the common cache entry of `cache_id`.
    pub use_cache_ids: Vec<Ident>,
}

/// Returns a mutable reference to the graph.
pub trait GraphReturn<IdxFrom: NodeIdx = GlobalCount> {
    /// Returns a reference to [`Graph`].
    fn graph(&self) -> Ref<Graph<IdxFrom>>;
    /// Returns a mutable reference to [`Graph`].
    fn graph_mut(&self) -> RefMut<Graph<IdxFrom>>;
}

/// Optimizes [`Graph`] and [`Cache`](crate::Cache) to achive a lower memory footprint.
#[cfg(feature = "opt-cache")]
pub trait GraphOpt {
    /// Optimizes [`Graph`] and [`Cache`](crate::Cache) to achive a lower memory footprint.
    fn optimize(&self) -> crate::Result<()>
    where
        Self: GraphReturn + CacheReturn + crate::PtrConv,
    {
        let mut cache = self.cache_mut();
        for trace in self.graph().cache_traces() {
            for node in &trace.use_cache_ids {
                // insert the common / optimized pointer in all the other nodes
                // this deallocates the old pointers
                let ptr = cache
                    .nodes
                    .get(&trace.cache_id)
                    .ok_or(DeviceError::GraphOptimization)?
                    .clone();
                cache.nodes.insert(*node, ptr);
            }
        }
        Ok(())
    }
}

#[cfg(feature = "opt-cache")]
impl<D: GraphReturn> GraphOpt for D {}


*/
