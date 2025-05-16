mod guard;
mod locked_array;
#[cfg(feature = "std")]
mod locked_map;
pub use guard::*;
pub use locked_array::*;
#[cfg(feature = "std")]
pub use locked_map::*;

pub type State<T> = Result<T, LockInfo>;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LockInfo {
    Locked,
    None,
}
