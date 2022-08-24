use custos::{get_device, number::Number, range, Buffer, Cache, GraphOpt, CPU};

pub trait AddBuf<T> {
    fn add(&self, lhs: &Buffer<T>, rhs: &Buffer<T>) -> Buffer<T>;
    fn relu(&self, lhs: &Buffer<T>) -> Buffer<T>;
}

impl<T> AddBuf<T> for CPU
where
    T: Number,
{
    fn add(&self, lhs: &Buffer<T>, rhs: &Buffer<T>) -> Buffer<T> {
        let len = std::cmp::min(lhs.len, rhs.len);

        let mut out = Cache::get(self, len, (lhs.node.idx, rhs.node.idx));

        for i in 0..len {
            out[i] = lhs[i] + rhs[i];
        }
        out
    }

    fn relu(&self, lhs: &Buffer<T>) -> Buffer<T> {
        let mut out = Cache::get(self, lhs.len, (lhs.node.idx, lhs.node.idx));

        for i in 0..lhs.len {
            if lhs[i] > T::zero() {
                out[i] = lhs[i];
            }
        }
        out
    }
}

pub trait AddOp<'a, T> {
    fn add(&self, rhs: &Buffer<'a, T>) -> Buffer<'a, T>;
    fn relu(&self) -> Buffer<'a, T>;
}

impl<'a, T: Number> AddOp<'a, T> for Buffer<'a, T> {
    #[inline]
    fn add(&self, rhs: &Buffer<'a, T>) -> Buffer<'a, T> {
        get_device!(self.device, AddBuf<T>).add(self, rhs)
    }

    fn relu(&self) -> Buffer<'a, T> {
        get_device!(self.device, AddBuf<T>).relu(self)
    }
}

#[test]
fn test_graph() {
    let device = CPU::new();

    // idx: 0
    let a = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));
    // idx: 1
    let b = Buffer::from((&device, [2, 3, 1, 4, 0, 5]));

    for ep in range(1) {
        // idx: 2, deps: [0, 1]
        let c = a.add(&b);
        assert_eq!(vec![3, 5, 4, 8, 5, 11], c.read());

        // idx: 3, deps: [2, 2]
        let d = c.relu();

        assert_eq!(vec![3, 5, 4, 8, 5, 11], d.read());

        // idx: 4, deps: [3, 1]
        let e = d.add(&b);

        if ep == 1 {
            assert_eq!(c.ptr, d.ptr);
            assert_eq!(c.ptr, e.ptr);
        }
        device.optimize();
    }
}
