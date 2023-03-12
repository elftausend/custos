/// A macro that creates a `CPU` `Buffer` using the static `CPU` device.
///
/// # Examples
#[cfg_attr(feature = "cpu", doc = "```")]
#[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
/// use custos::buf;
///
/// let buf = buf![2.; 10];
/// assert_eq!(buf.read(), [2.; 10]);
///
/// let buf = buf![5, 3, 2, 6, 2];
/// assert_eq!(buf.read(), &[5, 3, 2, 6, 2])
/// ```
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
        $crate::Buffer::<_, $crate::CPU, ()>::from([$($x),+])
    );

    // TODO: buf![device, [...]]
    ($device:expr, [($x:expr),+ $(,)?]) => (
        $crate::Buffer::<_, _, 0>::from((&device, [$($x),+]))
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_macro_filling() {
        let buf = buf![2.; 10];
        assert_eq!(buf.as_slice(), &[2.; 10]);
    }

    #[test]
    fn test_macro_from_slice() {
        let buf = buf![5, 3, 2, 6, 2];
        assert_eq!(buf.as_slice(), &[5, 3, 2, 6, 2])
    }
}
