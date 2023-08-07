use custos_macro::cuda;

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_macro() {
    use custos::{prelude::launch_kernel1d, Buffer, CUDA};

    // generic kernel
    let ptx = cuda!(
        r#"extern "C" __global__ void add(int* lhs, int* rhs, int* out, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx >= size) {
                return;
            }

            out[idx] = lhs[idx] + rhs[idx];
        }"#
    );

    println!("ptx: {ptx}");

    /*let ptx = r#"extern "C" __global__ void add(int* lhs, int* rhs, int* out, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= size) {
            return;
        }

        out[idx] = lhs[idx] + rhs[idx];
    }"#;*/
    let device = CUDA::new(0).unwrap();

    let lhs = Buffer::from((&device, [1, 2, 3, 4, 5]));
    let rhs = Buffer::from((&device, [1, 2, 3, 4, 5]));

    let mut out = Buffer::<i32, _>::new(&device, lhs.len());

    launch_kernel1d(
        lhs.len(),
        &device,
        ptx,
        "add",
        &[&lhs, &rhs, &mut out, &lhs.len()],
    )
    .unwrap();
}
