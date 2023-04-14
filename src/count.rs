use core::ops::{Range, RangeInclusive};

/// Converts ranges into a start and end index.
pub trait AsRangeArg {
    /// Returns the start index of the range.
    fn start(&self) -> usize;
    /// Returns the end index of the range.
    fn end(&self) -> usize;
}

impl AsRangeArg for Range<usize> {
    #[inline]
    fn start(&self) -> usize {
        self.start
    }

    #[inline]
    fn end(&self) -> usize {
        self.end
    }
}

impl AsRangeArg for RangeInclusive<usize> {
    #[inline]
    fn start(&self) -> usize {
        *self.start()
    }

    #[inline]
    fn end(&self) -> usize {
        *self.end() + 1
    }
}

impl AsRangeArg for usize {
    #[inline]
    fn start(&self) -> usize {
        0
    }

    #[inline]
    fn end(&self) -> usize {
        *self
    }
}

impl AsRangeArg for (usize, usize) {
    #[inline]
    fn start(&self) -> usize {
        self.0
    }

    #[inline]
    fn end(&self) -> usize {
        self.1
    }
}

/// used to reset the cache count in loops as every operation increases the cache count, which would break the "cache cycle" if the cache count would not be reset.
///
/// # Example
#[cfg_attr(not(feature = "no-std"), doc = "```")]
#[cfg_attr(feature = "no-std", doc = "```ignore")]
/// use custos::{get_count, range, Ident, bump_count};
///
/// for _ in range(100) { // using only one usize: exclusive range
///     Ident::new(10); // an 'Ident' is created if a Buffer is retrieved from cache.
///     bump_count();
///     assert!(get_count() == 1);
/// }
/// assert!(get_count() == 0);
/// ```
#[inline]
pub fn range<R: AsRangeArg>(range: R) -> Count {
    Count(range.start(), range.end())
}

/// used to reset the cache count
#[derive(Debug, Clone, Copy)]
pub struct Count(pub(super) usize, pub(super) usize);

/// The iterator used for setting the cache count.
#[derive(Debug)]
pub struct CountIntoIter {
    epoch: usize,
    #[cfg(not(feature = "no-std"))]
    idx: usize,
    end: usize,
}

impl Iterator for CountIntoIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        #[cfg(not(feature = "no-std"))]
        unsafe {
            crate::set_count(self.idx)
        };
        if self.epoch >= self.end {
            return None;
        }
        let epoch = Some(self.epoch);
        self.epoch += 1;
        epoch
    }
}

impl IntoIterator for Count {
    type Item = usize;

    type IntoIter = CountIntoIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        CountIntoIter {
            epoch: self.0,
            #[cfg(not(feature = "no-std"))]
            idx: crate::get_count(),
            end: self.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{range, Count, CountIntoIter};

    fn count_iter(iter: &mut CountIntoIter) {
        iter.next();
        assert_eq!(iter.epoch, 1);
        #[cfg(not(feature = "no-std"))]
        assert_eq!(iter.idx, 0);
        assert_eq!(iter.end, 10);

        iter.next();
        assert_eq!(iter.epoch, 2);
        #[cfg(not(feature = "no-std"))]
        assert_eq!(iter.idx, 0);
        assert_eq!(iter.end, 10);
    }

    #[test]
    fn test_count_into_iter() {
        let mut iter = CountIntoIter {
            epoch: 0,
            #[cfg(not(feature = "no-std"))]
            idx: 0,
            end: 10,
        };

        count_iter(&mut iter);
    }

    #[test]
    fn test_count() {
        let count: Count = Count(0, 10);
        count_iter(&mut count.into_iter());
    }

    #[test]
    fn test_range_inclusive() {
        let count: Count = range(0..=9);
        count_iter(&mut count.into_iter());

        for (idx, other) in count.into_iter().zip(0..=9) {
            assert_eq!(idx, other)
        }
    }
}
