//! Modules provide toggleable functionality for compute devices.
//! Modules are capable of swapping the used data structure used in [`Buffer`](crate::Buffer).
//! Custom modules can be created by implementing the [`Module`](crate::Module), [Setup](crate::Setup) traits and serveral pass down traits.
//! # Example
#![cfg_attr(feature = "cpu", doc = "```")]
#![cfg_attr(not(feature = "cpu"), doc = "```ignore")]
//! use custos::{Base, Module, Setup, CPU};
//!
//! pub struct CustomModule<Mods> {
//!     mods: Mods,
//! }
//!
//! impl<D, Mods: Module<D>> Module<D> for CustomModule<Mods> {
//!     type Module = CustomModule<Mods::Module>;
//!
//!     fn new() -> Self::Module {
//!         CustomModule {
//!             mods: Mods::new(),
//!         }
//!     }
//! }
//!
//! impl<Mods, D> Setup<D> for CustomModule<Mods>
//! where
//!     Mods: Setup<D>,
//! {
//!     fn setup(device: &mut D) -> custos::Result<()> {
//!         Mods::setup(device)
//!     }
//! }
//!
//! fn main() {
//!     let dev = CPU::<CustomModule<Base>>::new();
//!     // for actual usage, implement pass down traits / features
//! }
//! ```
//! The example above creates a custom module that wraps the [`Base``](crate::Base) module.

#[cfg(feature = "autograd")]
mod autograd;

#[cfg(feature = "autograd")]
pub use autograd::*;

mod base;
pub use base::*;

#[cfg(feature = "cached")]
mod cached;
#[cfg(feature = "cached")]
pub use cached::*;

#[cfg(feature = "graph")]
mod graph;
#[cfg(feature = "graph")]
pub use graph::*;

#[cfg(feature = "lazy")]
mod lazy;
#[cfg(feature = "lazy")]
pub use lazy::*;

#[cfg(feature = "fork")]
mod fork;
#[cfg(feature = "fork")]
pub use fork::*;

#[cfg(feature = "std")]
use crate::{Buffer, Device, HasId, Id, ShallowCopy, Shape, UniqueId};
#[cfg(feature = "std")]
use core::{any::Any, hash::BuildHasher};

#[cfg(feature = "std")]
use std::collections::HashMap;

pub trait Module<D, Mods = ()> {
    type Module;

    fn new() -> Self::Module;
}

#[cfg(feature = "std")]
#[inline]
#[allow(unused)]
pub(crate) unsafe fn register_buf_any<T, D, S>(
    cache: &mut HashMap<UniqueId, Box<dyn Any>, impl BuildHasher>,
    buf: &Buffer<T, D, S>,
) where
    T: 'static,
    D: Device + crate::IsShapeIndep + 'static,
    D::Data<T, S>: ShallowCopy,
    S: Shape,
{
    // shallow copy sets flag to AllocFlag::Wrapper

    let wrapped_data = buf.data.shallow();

    let buf = Buffer {
        data: wrapped_data,
        device: buf.device,
    };
    let buf: Buffer<'static, T, D, S> = core::mem::transmute(buf);
    cache.insert(*buf.id(), Box::new(buf));
}

#[cfg(feature = "std")]
#[inline]
#[allow(unused)]
pub(crate) fn unregister_buf_any(
    cache: &mut HashMap<UniqueId, Box<dyn Any>, impl BuildHasher>,
    id: Id,
) {
    cache.remove(&id);
}

#[cfg(feature = "std")]
#[inline]
#[allow(unused)]
pub(crate) unsafe fn register_buf_copyable<T, D, S>(
    cache: &mut HashMap<UniqueId, Box<dyn crate::BoxedShallowCopy>, impl BuildHasher>,
    buf: &Buffer<T, D, S>,
) where
    T: 'static,
    D: Device + crate::IsShapeIndep + 'static,
    D::Data<T, S>: ShallowCopy,
    S: Shape,
{
    // shallow copy sets flag to AllocFlag::Wrapper
    let wrapped_data = buf.data.shallow();

    let buf = Buffer {
        data: wrapped_data,
        device: buf.device,
    };
    let buf: Buffer<'static, T, D, S> = core::mem::transmute(buf);
    cache.insert(*buf.id(), Box::new(buf));
}

#[cfg(feature = "std")]
#[inline]
#[allow(unused)]
pub(crate) fn unregister_buf_copyable(
    cache: &mut HashMap<UniqueId, Box<dyn crate::BoxedShallowCopy>, impl BuildHasher>,
    id: Id,
) {
    cache.remove(&id);
}
