use custos::{
    prelude::Number,
    Buffer, CDatatype, OnDropBuffer, Shape, CUDA, cuda::launch_kernel,
};

pub fn cu_element_wise<Mods: OnDropBuffer, T: Number, S: Shape>(
    device: &CUDA<Mods>,
    lhs: &Buffer<T, CUDA<Mods>, S>,
    rhs: &Buffer<T, CUDA<Mods>, S>,
    out: &mut Buffer<T, CUDA<Mods>, S>,
    op: &str,
) -> custos::Result<()>
where
    T: CDatatype,
{
    let src = format!(
        r#"extern "C" __global__ void cu_ew({datatype}* lhs, {datatype}* rhs, {datatype}* out, int len) {{
            size_t idx = blockDim.x * blockIdx.x + threadIdx.x; 
            
            if (idx >= len) {{
                return;
            }}

            out[idx] = lhs[idx] {op} rhs[idx];
        }}"#,
        datatype = T::C_DTYPE_STR
    );

    launch_kernel(&device, [lhs.len() as u32 / 32, 1, 1], [32, 1, 1], 0, &src, "cu_ew", &[lhs, rhs, out, &lhs.len()])
    // device.launch_kernel1d(lhs.len(), src, "cu_ew", &[lhs, rhs, out, &lhs.len()])
}
mod tests {
    use custos::{CUDA, Base, prelude::{chosen_cu_idx}, Buffer, Retriever};


    const SIZE: usize = 655360;
    const TIMES: usize = 100;

    #[test]
    fn test_element_wise_large_bufs_cu() {
        use super::cu_element_wise;

        let device = CUDA::<Base>::new(chosen_cu_idx()).unwrap();

        let lhs = Buffer::from((&device, vec![1.0f32; SIZE]));
        let rhs = Buffer::from((&device, vec![4.0; SIZE]));

        let mut out = device.retrieve::<(), 0>(lhs.len(), ());

        let start = std::time::Instant::now();

        for _ in 0..TIMES {
            cu_element_wise::<_, _, ()>(&device, &lhs, &rhs, &mut out, "+").unwrap();
            // assert_eq!(out.read(), &[5.0; SIZE]);
        }

        println!("cu: {:?}", start.elapsed() /*/ TIMES as u32*/);

        assert_eq!(out.read(), &[5.0; SIZE]);

        println!("cu: {:?}", start.elapsed() /*/ TIMES as u32*/);
    }
}
