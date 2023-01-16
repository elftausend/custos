use custos::{Buffer, CacheAble2, Device, Shape, CPU};

type Return<'a, T = f32, D = CPU, S = ()> =
    <<D as Device>::Cache as CacheAble2<D>>::Retrieval<'a, T, S>;

pub trait AddBuf<T, S: Shape = ()>: Device {
    fn add(&self, lhs: &Buffer<T, Self, S>, rhs: &Buffer<T, Self, S>) -> Return<'_, T, Self, S>;
}

pub struct Test<'a, T: 'a, D: Device + 'a> {
    val: Return<'a, T, D>,
}

impl<T, S> AddBuf<T, S> for CPU
where
    T: Copy + std::ops::Add<Output = T>,
    S: Shape,
{
    fn add(&self, lhs: &Buffer<T, Self, S>, rhs: &Buffer<T, Self, S>) -> Return<'_, T, Self, S> {
        let len = std::cmp::min(lhs.len(), rhs.len());
        let out = self.retrieve(len, (lhs, rhs));

        for i in 0..len {
            out[i] = lhs[i] + rhs[i];
        }

        out
    }
}

fn main() {
    let device = CPU::new();

    let lhs = Buffer::from((&device, [3, 6, 4, 1, 7]));
    let rhs = Buffer::from((&device, [2, 1, 6, 3, 2]));

    let out = device.add(&lhs, &rhs);
    println!("out: {out:?}");
}
