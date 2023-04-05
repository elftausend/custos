use core::{
    cell::{Ref, RefCell, RefMut},
    fmt::Debug,
};

use crate::{
    keeper::Keeper, Cache, CacheReturn, Device, GlobalCount, Graph, GraphReturn, NodeIdx, PtrConv,
};

/// Provides several addons for a device.
/// - `graph`: An optimizeable graph.
/// - `cache`: A cache for allocations.
/// - `tape`: A (gradient) tape.
pub struct Addons<D: Device, IdxFrom: NodeIdx = GlobalCount> {
    pub graph: RefCell<Graph<IdxFrom>>,
    pub cache: RefCell<Cache<D>>,

    #[cfg(feature = "autograd")]
    pub keeper: RefCell<Keeper<D>>,
    #[cfg(feature = "autograd")]
    pub tape: RefCell<crate::Tape<D>>,
}

impl<D: Device + Debug> Debug for Addons<D>
where
    D::Ptr<u8, ()>: Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        #[cfg(feature = "autograd")]
        {
            f.debug_struct("Addons")
                .field("graph", &self.graph)
                .field("cache", &self.cache)
                .field("tape", &self.tape)
                .finish()
        }

        #[cfg(not(feature = "autograd"))]
        f.debug_struct("Addons")
            .field("graph", &self.graph)
            .field("cache", &self.cache)
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

            #[cfg(feature = "autograd")]
            keeper: Default::default(),
            #[cfg(feature = "autograd")]
            tape: Default::default(),
        }
    }
}

impl<D: GraphReturn + Device + PtrConv> Addons<D> {}

/// `AddonsReturn` is probably implemented for all devices that have an [`Addons`] field.
pub trait AddonsReturn: Device {
    /// Returns a reference to [`Addons`].
    fn addons(&self) -> &Addons<Self>;
}

impl<D: AddonsReturn> GraphReturn for D {
    #[inline]
    fn graph(&self) -> Ref<Graph<GlobalCount>> {
        self.addons().graph.borrow()
    }

    #[inline]
    fn graph_mut(&self) -> RefMut<Graph<GlobalCount>> {
        self.addons().graph.borrow_mut()
    }
}

impl<D: AddonsReturn> CacheReturn for D {
    #[inline]
    fn cache(&self) -> Ref<crate::Cache<Self>>
    where
        Self: PtrConv,
    {
        self.addons().cache.borrow()
    }

    #[inline]
    fn cache_mut(&self) -> RefMut<crate::Cache<Self>>
    where
        Self: PtrConv,
    {
        self.addons().cache.borrow_mut()
    }
}

#[cfg(feature = "autograd")]
impl<D: AddonsReturn> crate::TapeReturn for D {
    #[inline]
    fn tape(&self) -> Ref<crate::Tape<Self>> {
        self.addons().tape.borrow()
    }

    #[inline]
    fn tape_mut(&self) -> RefMut<crate::Tape<Self>> {
        self.addons().tape.borrow_mut()
    }
}

/// This trait is implemented for all devices that provide a [`Keeper`].
pub trait KeeperReturn: Device {
    fn keeper(&self) -> Ref<Keeper<Self>>;
    fn keeper_mut(&self) -> RefMut<Keeper<Self>>;
}

#[cfg(feature = "autograd")]
impl<D: AddonsReturn> KeeperReturn for D {
    #[inline]
    fn keeper(&self) -> Ref<Keeper<Self>> {
        self.addons().keeper.borrow()
    }

    #[inline]
    fn keeper_mut(&self) -> RefMut<Keeper<Self>> {
        self.addons().keeper.borrow_mut()
    }
}
