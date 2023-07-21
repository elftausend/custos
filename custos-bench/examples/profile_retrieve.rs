use std::hint::black_box;

use custos::module_comb::{Cached, Base, Retriever};

const SIZE: usize = 10;

fn main() {
    let device = custos::module_comb::CPU::<Cached<Base>>::new();

    for _ in 0..100 {
        let _out = black_box(device.retrieve::<f32, ()>(SIZE));
        let _out = black_box(device.retrieve::<f32, ()>(SIZE));
        let _out = black_box(device.retrieve::<f32, ()>(SIZE));
        let _out = black_box(device.retrieve::<f32, ()>(SIZE));
        let _out = black_box(device.retrieve::<f32, ()>(SIZE));
        let _out = black_box(device.retrieve::<f32, ()>(SIZE));
        let _out = black_box(device.retrieve::<f32, ()>(SIZE));
        let _out = black_box(device.retrieve::<f32, ()>(SIZE));
        let _out = black_box(device.retrieve::<f32, ()>(SIZE));
        let _out = black_box(device.retrieve::<f32, ()>(SIZE));
    }
    
}