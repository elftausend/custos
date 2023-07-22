use std::hint::black_box;

use custos::module_comb::{Base, Cached, Retriever};

const SIZE: usize = 10;

fn main() {
    let device = custos::module_comb::CPU::<Cached<Base>>::new();

    let start = std::time::Instant::now();

    for _ in 0..10000 {
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

    println!("dur: {:?}", start.elapsed() / 10000);
}
