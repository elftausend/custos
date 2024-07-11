mod location_hasher;
pub use location_hasher::*;

mod nohasher;
pub use nohasher::*;

mod any_buffer;

#[cfg(feature = "cached")]
mod borrow_cache;
#[cfg(feature = "cached")]
pub use borrow_cache::*;

#[cfg(feature = "cached")]
mod owned_cache;
#[cfg(feature = "cached")]
pub use owned_cache::*;

pub type UniqueId = u64;
