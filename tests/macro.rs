#[cfg(feature = "cuda")]
use custos::prelude::*;
#[cfg(feature = "cuda")]
#[cfg(feature = "macro")]
use custos_macro::cuda;

#[cfg(feature = "macro")]
#[cfg(feature = "cuda")]
#[test]
fn test_cuda_macro() {
    // let ptx = cuda!(
    //     template<typename T>
    //     __global__ void add(T* lhs, T* rhs, T* out, int size) {
    //         int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //         if (idx >= size) {
    //             return;
    //         }

    //         out[idx] = lhs[idx] + rhs[idx];
    //     }
    // );

    // println!("ptx: {ptx}");
    // return;

    // generic kernel
    /*let ptx = cuda!(
        extern "C" __global__ void add(int* lhs, int* rhs, int* out, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx >= size) {
                return;
            }

            out[idx] = lhs[idx] + rhs[idx];
        }
    );

    println!("ptx: {ptx}");*/

    /*let ptx = r#"extern "C" __global__ void add(int* lhs, int* rhs, int* out, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= size) {
            return;
        }

        out[idx] = lhs[idx] + rhs[idx];
    }"#;*/
    let device = CUDA::<Base>::new(0).unwrap();

    let lhs = Buffer::from((&device, [1, 2, 3, 4, 5]));
    let rhs = Buffer::from((&device, [1, 2, 3, 4, 5]));

    let mut out = Buffer::<i32, _>::new(&device, lhs.len());

    // device
    //     .launch_kernel1d(lhs.len(), ptx, "add", &[&lhs, &rhs, &mut out, &lhs.len()])
    //     .unwrap();*
}
