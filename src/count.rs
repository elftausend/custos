use std::ops::Range;

use crate::{get_count, set_count};

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
/// used to reset the cache count in loops as every operation increases the cache count, which would break the "cache cycle".
///
/// # Example
/// ```
/// use custos::{get_count, range, Node};
///
/// for _ in range(100) {
///     Node::new(10); // a 'Node' is created if a Buffer is retrieved from cache.
///     assert!(get_count() == 1);
/// }
/// assert!(get_count() == 0);
/// ```
pub fn range<R: AsRangeArg>(range: R) -> Count {
    Count(range.start(), range.end())
}

/// used to reset the cache count
pub struct Count(usize, usize);

pub struct CountIntoIter {
    epoch: usize,
    idx: usize,
    end: usize,
}

impl Iterator for CountIntoIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        set_count(self.idx);
        if self.epoch > self.end {
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
            idx: get_count(),
            end: self.1,
        }
    }
}
