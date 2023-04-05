pub(crate) mod api;

/// Determines the order of stored elements in a matrix
#[repr(C)]
pub enum Order {
    /// Matrix is stored in row-major order
    RowMajor = 101,
    /// Matrix is stored in column-major order
    ColMajor = 102,
}

/// Wrapper for bare numbers that determines whether an input should be transposed or not
#[repr(C)]
pub enum Transpose {
    /// Do not tranpose input matrix
    NoTrans = 111,
    /// Transpose input matrix
    Trans = 112,
}
