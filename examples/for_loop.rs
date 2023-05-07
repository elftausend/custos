use custos::prelude::*;

fn main() {
    let device = CPU::new();

    // `range` resets the cache count in every iteration.
    // The cache count is used to retrieve the same allocation in each iteration.
    // Not adding `range` results in allocating new memory in each iteration,
    // which is only freed when the device is dropped.
    // To disable this caching behaviour, the `realloc` feature can be enabled.
    for _ in range(5) {
        // uses the same allocation in each iteration
        let _buf = device.retrieve::<f32, ()>(10, ());
    }
}
