
pub fn assign_to_lhs<T: Copy, F>(lhs: &mut [T], rhs: &[T], f: F) where F: Fn(&mut T, T) 
{
    let len = core::cmp::min(lhs.len(), rhs.len());
    
    let rhs = &rhs[..len];
    let lhs_slice = &mut lhs[..len];

    for idx in 0..len {
        f(&mut lhs_slice[idx], rhs[idx])
    }
}