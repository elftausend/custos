use custos::{prelude::*, WrappedData};

fn scalar_apply<'a, Mods: Retrieve<CUDA<Mods>, f32>>(
    device: &'a CUDA<Mods>,
    lhs: &'a Buffer<f32, CUDA<Mods>>,
    rhs: f32,
) -> custos::Result<Buffer<'a, f32, CUDA<Mods>>> {
    let src = r#"extern "C" __global__ void scalar_add(float* lhs, float rhs, float* out, int numElements)
            {
                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                if (idx < numElements) {
                    out[idx] = lhs[idx] + rhs;
                }
              
            }
    "#;

    let out: Buffer<f32, _> = device.retrieve(lhs.len(), ()).unwrap();
    device.launch_kernel1d(
        lhs.len(),
        src,
        "scalar_add",
        &[&lhs, &rhs, &out, &lhs.len()],
    )?;
    Ok(out)
}

#[test]
fn test_scalar_op_cuda() -> custos::Result<()> {
    let device = CUDA::<Base>::new(0)?;

    let lhs = Buffer::from((&device, [1f32, 2., 3., 4., 5.]));

    let out = scalar_apply(&device, &lhs, 3.)?;

    assert_eq!(vec![4., 5., 6., 7., 8.], out.read());
    Ok(())
}

#[cfg(feature = "static-api")]
#[test]
fn test_large_scalar_ops_cuda_static_api() -> custos::Result<()> {
    use custos::static_api::{static_cuda, Mods};

    let lhs = (0..100000).into_iter()
        .map(|val| val as f32)
        .collect::<Buffer::<f32, CUDA<Mods<CUDA>>>>()/*.to_cuda() */;

    let out = scalar_apply(&static_cuda(), &lhs, 1.)?;

    let actual = (0..100000)
        .into_iter()
        .map(|val| val as f32 + 1.)
        .collect::<Vec<f32>>();
    assert_eq!(actual, out.read());

    Ok(())
}
