use crate::{HasId, PtrType, ShallowCopy, flag::AllocFlag};
use nnapi::Operand;

#[derive(Debug, Clone)]
/// Denotes an index with a data type to a node in a nnapi model
pub struct NnapiPtr {
    /// The data type of the node
    pub dtype: Operand,
    /// The index of the node
    pub idx: u32,
    pub flag: AllocFlag,
}

impl Default for NnapiPtr {
    fn default() -> Self {
        Self {
            dtype: Operand::activation(),
            idx: u32::MAX,
            flag: AllocFlag::Wrapper,
        }
    }
}

impl HasId for NnapiPtr {
    #[inline]
    fn id(&self) -> crate::Id {
        crate::Id {
            id: self.idx as u64,
            len: self.dtype.len,
        }
    }
}

impl PtrType for NnapiPtr {
    #[inline]
    fn size(&self) -> usize {
        self.dtype.len
    }

    #[inline]
    fn flag(&self) -> crate::flag::AllocFlag {
        self.flag
    }

    #[inline]
    unsafe fn set_flag(&mut self, flag: AllocFlag) {
        self.flag = flag;
    }
}

impl ShallowCopy for NnapiPtr {
    unsafe fn shallow(&self) -> Self {
        NnapiPtr {
            dtype: self.dtype.clone(),
            idx: self.idx,
            flag: crate::flag::AllocFlag::Wrapper,
        }
    }
}
