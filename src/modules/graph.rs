mod graph_translator;
mod node;
mod opt_graph;

pub use node::Node;
pub use opt_graph::*;

use core::{cell::RefCell, panic::Location};

use crate::{
    pass_down_add_operation, pass_down_exec_now_module, pass_down_unified_mem_chain,
    pass_down_use_gpu_or_cpu, Alloc, Buffer, Device, HasId, Module, OnDropBuffer, OnNewBuffer,
    OptimizeMemGraph, Parents, PtrConv, Retrieve, Setup, Shape, TranslatedCacheTrace,
};

use self::graph_translator::GraphTranslator;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Graph<Mods> {
    pub modules: Mods,
    pub graph_trans: RefCell<GraphTranslator>,
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

impl<Mods: OptimizeMemGraph> OptimizeMemGraph for Graph<Mods> {
    fn optimize_mem_graph(
        &self,
        cache_traces: Option<&[TranslatedCacheTrace]>,
    ) -> crate::Result<()> {
        match cache_traces {
            Some(cache_traces) => self.modules.optimize_mem_graph(Some(cache_traces)),
            None => {
                let graph_trans = self.graph_trans.borrow();
                let idx_to_loc = &graph_trans.idx_to_buf_location;
                let cache_traces = graph_trans.opt_graph.cache_traces();

                let cache_traces = cache_traces
                    .into_iter()
                    .map(|cache_trace| TranslatedCacheTrace {
                        cache_idx: *idx_to_loc.get(&cache_trace.cache_idx).unwrap(),
                        use_cache_idxs: cache_trace
                            .use_cache_idxs
                            .into_iter()
                            .map(|cache_idx| *idx_to_loc.get(&cache_idx).unwrap())
                            .collect(),
                    })
                    .collect::<Vec<_>>();

                self.modules.optimize_mem_graph(Some(&cache_traces))
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
        graph_trans
            .idx_to_buf_location
            .insert(next_idx, Location::caller().into());

        // does a hash location check internally (again)
        graph_trans.add_node(len, &ids);
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
