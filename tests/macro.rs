use custos_macro::cuda;

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_macro() {
    let ptx = cuda!(
        template <class T>
        __global__ void add(T* lhs, T* rhs, T* out, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx >= size) {
                return;
            }

            out[idx] = lhs[idx] + rhs[idx];
        }
    );
    println!("ptx: {ptx}");
}