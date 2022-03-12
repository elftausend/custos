use std::ops::Range;

use crate::libs::opencl::CLCACHE_COUNT;


pub fn range(range: Range<usize>) -> Count {
    Count(range.start, range.end)
}

pub struct Count(usize, usize);

pub struct CountIntoIter {
    epoch: usize,
    idx: usize,
    end: usize,
}

impl Iterator for CountIntoIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe { CLCACHE_COUNT = self.idx };
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
            idx: unsafe { CLCACHE_COUNT },
            end: self.1
        }
    }
}
