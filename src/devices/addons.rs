use core::{cell::RefCell, fmt::Debug};

use crate::{Cache, CacheReturn, Device, GlobalCount, Graph, GraphReturn, NodeIdx, RawConv};

/// Provides several addons for a device.
/// - `graph`: An optimizeable graph.
/// - `cache`: A cache for allocations.
/// - `tape`: A (gradient) tape.
#[derive(Debug, Default)]
pub struct Addons<D: RawConv, IdxFrom: NodeIdx = GlobalCount> {
    pub graph: RefCell<Graph<IdxFrom>>,
    pub cache: RefCell<Cache<D>>,
    #[cfg(feature = "autograd")]
    pub tape: RefCell<crate::Tape<D>>,
}

/// `AddonsReturn` is probably implemented for all devices that have an [`Addons`] field.
pub trait AddonsReturn: Device + RawConv {
    /// The pointer type used for the [`Cache`].
    type CachePtrType: Debug;

    /// Returns a reference to [`Addons`].
    fn addons(&self) -> &Addons<Self>;
}

impl<D: AddonsReturn> GraphReturn for D {
    #[inline]
    fn graph(&self) -> std::cell::Ref<Graph<GlobalCount>> {
        self.addons().graph.borrow()
    }

    #[inline]
    fn graph_mut(&self) -> std::cell::RefMut<Graph<GlobalCount>> {
        self.addons().graph.borrow_mut()
    }
}

impl<D: AddonsReturn> CacheReturn for D {
    type CT = D::CachePtrType;

    #[inline]
    fn cache(&self) -> core::cell::Ref<crate::Cache<Self>>
    where
        Self: crate::RawConv,
    {
        self.addons().cache.borrow()
    }

    #[inline]
    fn cache_mut(&self) -> core::cell::RefMut<crate::Cache<Self>>
    where
        Self: crate::RawConv,
    {
        self.addons().cache.borrow_mut()
    }
}

#[cfg(feature = "autograd")]
impl<D: AddonsReturn> crate::TapeReturn for D {
    #[inline]
    fn tape(&self) -> core::cell::Ref<crate::Tape<Self>> {
        self.addons().tape.borrow()
    }

    #[inline]
    fn tape_mut(&self) -> core::cell::RefMut<crate::Tape<Self>> {
        self.addons().tape.borrow_mut()
    }
}
