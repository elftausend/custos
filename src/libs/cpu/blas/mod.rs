pub mod api;

#[repr(C)]
pub enum Order {
    RowMajor = 101,
    ColMajor = 102,
}
#[repr(C)]
pub enum Transpose {
    NoTranspose = 111,
    Transpose = 112,
}
