use custos::Buffer;

fn main() {
    let mut buf = Buffer::from([1, 2, 3, 6, 5, 3, -4]); // or:
    // let mut buf = custos::buf![4, 3, 4, 4];

    for value in &mut buf {
        *value -= 2;
    }

    let mut gpu_buf = buf.to_gpu();

    assert_eq!(gpu_buf.read(), [-1, 0, 1, 4, 3, 1, -6]);

    gpu_buf.clear();

    assert_eq!(gpu_buf.read(), [0; 7]);
}
