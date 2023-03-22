use core::{cell::RefCell, fmt::Debug};

use crate::{Cache, CacheReturn, Device, GlobalCount, Graph, GraphReturn, NodeIdx, PtrConv};

/// Provides several addons for a device.
/// - `graph`: An optimizeable graph.
/// - `cache`: A cache for allocations.
/// - `tape`: A (gradient) tape.
pub struct Addons<D: Device, IdxFrom: NodeIdx = GlobalCount> {
    pub graph: RefCell<Graph<IdxFrom>>,
    pub cache: RefCell<Cache<D>>,
    #[cfg(feature = "autograd")]
    pub tape: RefCell<crate::Tape<D>>,
}

impl<D: Device + Debug> Debug for Addons<D>
where
    D::Ptr<u8, ()>: Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Addons")
            .field("graph", &self.graph)
            .field("cache", &self.cache)
            .field("tape", &self.tape)
            .finish()
    }
}

impl<D: Device + Default> Default for Addons<D>
where
    D::Ptr<u8, ()>: Default,
{
    fn default() -> Self {
        Self {
            graph: Default::default(),
            cache: Default::default(),
            tape: Default::default(),
        }
    }
}

/// `AddonsReturn` is probably implemented for all devices that have an [`Addons`] field.
pub trait AddonsReturn: Device {
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
    #[inline]
    fn cache(&self) -> core::cell::Ref<crate::Cache<Self>>
    where
        Self: PtrConv,
    {
        self.addons().cache.borrow()
    }

    #[inline]
    fn cache_mut(&self) -> core::cell::RefMut<crate::Cache<Self>>
    where
        Self: PtrConv,
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
