#[derive(Debug, Clone)]
pub struct Locked<T> {
    locked: bool,
    data: T,
}

impl<T> Locked<T> {
    pub fn new(data: T) -> Self {
        Self {
            locked: false,
            data,
        }
    }
    pub fn data(&self) -> Option<&T> {
        if self.locked {
            Some(&self.data)
        } else {
            None
        }
    }
}
