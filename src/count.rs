use core::ops::{Range, RangeInclusive};

pub trait AsRangeArg {
    fn start(&self) -> usize;
    fn end(&self) -> usize;
}

impl AsRangeArg for Range<usize> {
    fn start(&self) -> usize {
        self.start
    }

    fn end(&self) -> usize {
        self.end
    }
}

impl AsRangeArg for RangeInclusive<usize> {
    fn start(&self) -> usize {
        *self.start()
    }

    fn end(&self) -> usize {
        *self.end() + 1
    }
}

impl AsRangeArg for usize {
    fn start(&self) -> usize {
        0
    }

    fn end(&self) -> usize {
        *self
    }
}

impl AsRangeArg for (usize, usize) {
    fn start(&self) -> usize {
        self.0
    }

    fn end(&self) -> usize {
        self.1
    }
}

/// inclusive range
/// used to reset the cache count in loops as every operation increases the cache count, which would break the "cache cycle" if the cache count would not be reset.
///
/// # Example
/// ```
/// use custos::{get_count, range, Ident, bump_count};
///
/// for _ in range(100) {
///     Ident::new(10); // an 'Ident' is created if a Buffer is retrieved from cache.
///     bump_count();
///     assert!(get_count() == 1);
/// }
/// assert!(get_count() == 0);
/// ```
pub fn range<R: AsRangeArg>(range: R) -> Count {
    Count(range.start(), range.end())
}

/// used to reset the cache count
#[derive(Debug, Clone, Copy)]
pub struct Count(pub(super) usize, pub(super) usize);

#[derive(Debug)]
pub struct CountIntoIter {
    epoch: usize,
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

    fn into_iter(self) -> Self::IntoIter {
        CountIntoIter {
            epoch: self.0,
            #[cfg(not(feature = "no-std"))]
            idx: crate::get_count(),
            #[cfg(feature = "no-std")]
            idx: 0,
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
        assert_eq!(iter.idx, 0);
        assert_eq!(iter.end, 10);

        iter.next();
        assert_eq!(iter.epoch, 2);
        assert_eq!(iter.idx, 0);
        assert_eq!(iter.end, 10);
    }

    #[test]
    fn test_count_into_iter() {
        let mut iter = CountIntoIter {
            epoch: 0,
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
