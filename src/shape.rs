
pub trait Shape {
    const LEN: usize = 0;
    type ARR<T>;
}

impl Shape for () {
    type ARR<T> = ();
}

pub struct Dim1<const N: usize> {}

impl<const N: usize> Shape for Dim1<N> {
    const LEN: usize = N;
    type ARR<T> = [T; N];
}

pub struct Dim2<const B: usize, const A: usize> {}

impl<const B: usize, const A: usize> Shape for Dim2<B, A> {
    const LEN: usize = B*A;
    type ARR<T> = [[T; A]; B];
}

pub struct Dim3<const C: usize, const B: usize, const A: usize> {}

impl<const C: usize, const B: usize, const A: usize> Shape for Dim3<C, B, A> {
    const LEN: usize = B*A*C;
    type ARR<T> = [[[T; A]; B]; C];
}

