mod locked_array;
mod guard;
pub use guard::*;
pub use locked_array::*;

pub type State<T> = Result<T, LockInfo>;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LockInfo {
    Locked,
    None,
}