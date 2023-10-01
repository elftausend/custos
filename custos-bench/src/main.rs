use std::hint::black_box;

use custos::{Base, Buffer, Cached, Retriever};

const SIZE: usize = 10;

fn main() {
    let device = custos::CPU::<Cached<Base>>::new();

    let start = std::time::Instant::now();

    for _ in 0..10000 {
        let _out: Buffer<f32, _, _> = black_box(device.retrieve::<(), 0>(SIZE, ()));
        let _out: Buffer<f32, _, _> = black_box(device.retrieve::<(), 0>(SIZE, ()));
        let _out: Buffer<f32, _, _> = black_box(device.retrieve::<(), 0>(SIZE, ()));
        let _out: Buffer<f32, _, _> = black_box(device.retrieve::<(), 0>(SIZE, ()));
        let _out: Buffer<f32, _, _> = black_box(device.retrieve::<(), 0>(SIZE, ()));
        let _out: Buffer<f32, _, _> = black_box(device.retrieve::<(), 0>(SIZE, ()));
        let _out: Buffer<f32, _, _> = black_box(device.retrieve::<(), 0>(SIZE, ()));
        let _out: Buffer<f32, _, _> = black_box(device.retrieve::<(), 0>(SIZE, ()));
        let _out: Buffer<f32, _, _> = black_box(device.retrieve::<(), 0>(SIZE, ()));
        let _out: Buffer<f32, _, _> = black_box(device.retrieve::<(), 0>(SIZE, ()));
    }

    println!("dur: {:?}", start.elapsed() / 10000);
}
