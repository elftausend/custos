pub mod api;

#[repr(C)]
pub enum Order {
    RowMajor = 101,
    ColMajor = 102,
}
#[repr(C)]
pub enum Transpose {
    NoTrans = 111,
    Trans = 112,
}
