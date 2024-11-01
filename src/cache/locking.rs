mod guard;
mod locked_array;
mod locked_map;
pub use guard::*;
pub use locked_array::*;
pub use locked_map::*;

pub type State<T> = Result<T, LockInfo>;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LockInfo {
    Locked,
    None,
}
