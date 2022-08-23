use custos::{get_device, number::Number, range, Buffer, Cache, GraphReturn, CPU};

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

        let mut out = Cache::get(self, len, lhs.node.idx, rhs.node.idx);

        for i in 0..len {
            out[i] = lhs[i] + rhs[i];
        }
        out
    }

    fn relu(&self, lhs: &Buffer<T>) -> Buffer<T> {
        let mut out = Cache::get(self, lhs.len, lhs.node.idx, lhs.node.idx);

        for i in 0..lhs.len {
            if out[i] > T::zero() {
                out[i] = T::zero();
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

    // idx: 2, deps: [0, 1]
    let c = a.add(&b);

    // idx: 3, deps: [2, 2]
    let d = c.relu();
    // idx: 4, deps: [3, 1]
    let e = d.add(&b);

    // idx: 5, deps: [2, 1]
    let f = c.add(&b);

    let mut graph = device.graph();
    //println!("graph: {graph:?}");
    let is_c_opt = graph.is_path_optimizable(&c.node);
    //let is_opt = graph.is_optimizable();
    println!("is_opt: {is_c_opt}");
    //println!("c: {c:?}");
}
