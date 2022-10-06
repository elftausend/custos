
#[macro_export]
macro_rules! buf {
    ($elem:expr; $n:expr) => (
        if $n == 0 {
            panic!("The length of the buffer can't be 0.");
        } else {
            $crate::Buffer::from(vec![$elem; $n])
        }
    );

    ($($x:expr),+ $(,)?) => (
        $crate::Buffer::from(vec![$($x),+])
    )
}