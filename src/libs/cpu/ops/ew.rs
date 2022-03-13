use alloc::vec::Vec;

extern crate alloc;

pub fn element_wise_op<T: Copy+Default, F>(lhs: &[T], rhs: &[T], f: F) -> Vec<T> 
where F: Fn(T, T) -> T {
    
    let len = core::cmp::min(lhs.len(), rhs.len());
    let lhs = &lhs[..len];
    let rhs = &rhs[..len];

    let mut vec = vec![T::default(); len];
    
    let out_slice = &mut vec[..len];

    for idx in 0..len {
        out_slice[idx] = f(lhs[idx], rhs[idx])
    }
    vec
}

pub fn element_wise_op_mut<T: Copy, F>(lhs: &[T], rhs: &[T], out: &mut [T], f: F) where F: Fn(T, T) -> T 
{
    
    let len = core::cmp::min(lhs.len(), rhs.len());
    let lhs = &lhs[..len];
    let rhs = &rhs[..len];

    let out_slice = &mut out[..len];

    for idx in 0..len {
        out_slice[idx] = f(lhs[idx], rhs[idx])
    }

}