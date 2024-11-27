use custos::{Base, Buffer, CUDA, Device};

pub struct State<'a, 'b> {
    device: &'a CUDA,
    lhs: &'a Buffer<'b, i32, CUDA>,
    rhs: &'a Buffer<'b, i32, CUDA>,
    out: &'a Buffer<'b, i32, CUDA>,
}

pub struct Action<H> {
    handler: H,
}

pub fn call() {}

pub fn sum_kernel(device: &CUDA, x: &Buffer<i32, CUDA>, out: &mut Buffer<i32, CUDA>) {
    let src = r#"
        extern "C" __global__ void countZeros(int *d_A, int* B, int numElements)
            {
                int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < numElements) {                    
                    atomicAdd(&(B[0]), d_A[index]); 
                }
            }
    "#;

    device
        .launch_kernel1d(x.len(), src, "countZeros", &[x, out, &x.len()])
        .unwrap();
}

const N: usize = 20000;

#[test]
fn test_cuda_sum() {
    let device = CUDA::<Base>::new(0).unwrap();
    let lhs = Buffer::from((&device, 0..N));
    let mut out = Buffer::from((&device, [0]));
    sum_kernel(&device, &lhs, &mut out);
    out.clear();
    let start = std::time::Instant::now();
    sum_kernel(&device, &lhs, &mut out);

    assert_eq!(out.read(), [(N as i128 * (N as i128 - 1) / 2) as i32]);
    println!("Time: {:?}", start.elapsed());
}

// find which one is faster -> then use that one dynamically, run in async

#[test]
fn test_cuda_sum_two() {
    let device = CUDA::<Base>::new(0).unwrap();
    let lhs = Buffer::<_, _>::from((&device, (0..N)));
    // println!("lhs: {:?}", lhs);

    let start = std::time::Instant::now();

    assert_eq!(
        lhs.read().iter().sum::<i32>() as i128,
        N as i128 * (N as i128 - 1) / 2
    );
    println!("Time: {:?}", start.elapsed());
}
