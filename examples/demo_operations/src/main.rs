mod linear_layer;

use custos::{Buffer, CacheAble2, Device, Shape, CPU, Stack};
use custos_macro::impl_stack;

type Return<'a, T = f32, D = CPU, S = ()> =
    <<D as Device>::Cache as CacheAble2<D>>::Retrieval<'a, T, S>;

pub trait AddBuf<'a, T, S: Shape = ()>: Device {
    fn add(&'a self, lhs: &Buffer<T, Self, S>, rhs: &Buffer<T, Self, S>) -> Return<'a, T, Self, S>;
}

pub struct Test<'a, T: 'a, D: Device + 'a> {
    val: Option<Return<'a, T, D>>,
}

//#[impl_stack]
impl<'a, T, S> AddBuf<'a, T, S> for CPU
where
    T: Copy + std::ops::Add<Output = T>,
    S: Shape,
{
    fn add(&'a self, lhs: &Buffer<T, Self, S>, rhs: &Buffer<T, Self, S>) -> Return<'a, T, Self, S> {
        let len = std::cmp::min(lhs.len(), rhs.len());
        let out = self.retrieve(len, (lhs, rhs));

        for i in 0..len {
            out[i] = lhs[i] + rhs[i];
        }

        out
    }
}

impl<'a, T, S> AddBuf<'a, T, S> for Stack
where
    T: Copy + std::ops::Add<Output = T>+ 'a,
    S: Shape +'a
{
    fn add(&self, lhs: &Buffer<T, Self, S>, rhs: &Buffer<T, Self, S>) -> Return<'a, T, Self, S> {
        todo!()
    }
}


fn main() {
    let device = CPU::new();
    //let device = custos::Stack;

    let lhs = Buffer::from((&device, [3, 6, 4, 1, 7]));
    let rhs = Buffer::from((&device, [2, 1, 6, 3, 2]));

    let mut test: Test<i32, CPU> = Test {
        val: None,
    };

    for _ in 0..100 {
        let out = device.add(&lhs, &rhs);
        println!("out: {out:?}");
        test.val = Some(out);
    }
    
    
}
