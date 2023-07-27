#[derive(Default)]
pub struct NoHasher {
    hash: u64,
}

impl std::hash::Hasher for NoHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }

    #[inline]
    fn write(&mut self, _bytes: &[u8]) {
        unimplemented!("NoHasher only hashes u64, (u32 and usize as u64 cast).")
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.hash = i;
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.write_u64(i as u64);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }
}
