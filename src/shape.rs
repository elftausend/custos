pub unsafe trait Shape {
    const LEN: usize = 0;
    type ARR<T>;

    fn new<T: Copy + Default>() -> Self::ARR<T>;
}

unsafe impl Shape for () {
    type ARR<T> = ();

    fn new<T>() -> Self::ARR<T> {
        ()
    }
}

pub struct Dim1<const N: usize> {}

unsafe impl<const N: usize> Shape for Dim1<N> {
    const LEN: usize = N;
    type ARR<T> = [T; N];

    fn new<T: Copy + Default>() -> Self::ARR<T> {
        [T::default(); N]
    }
}

pub struct Dim2<const B: usize, const A: usize> {}

unsafe impl<const B: usize, const A: usize> Shape for Dim2<B, A> {
    const LEN: usize = B * A;
    type ARR<T> = [[T; A]; B];

    fn new<T: Copy + Default>() -> Self::ARR<T> {
        [[T::default(); A]; B]
    }
}

pub struct Dim3<const C: usize, const B: usize, const A: usize> {}

unsafe impl<const C: usize, const B: usize, const A: usize> Shape for Dim3<C, B, A> {
    const LEN: usize = B * A * C;
    type ARR<T> = [[[T; A]; B]; C];

    fn new<T: Copy + Default>() -> Self::ARR<T> {
        [[[T::default(); A]; B]; C]
    }
}
