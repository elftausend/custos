//use custos::{opencl::KernelOptions, CLDevice, Error, AsDev, Matrix};


/* 
#[test]
fn test_cl_offset() -> Result<(), Error> {
    let device = CLDevice::get(0)?.select();

    let lhs = Matrix::from((&device, (2, 3), 
        [1., 2., 3., 
        4., 5., 6.,]
    ));
    let rhs = Matrix::from((&device, (2, 3), 
        [0.5, 1., 1.5, 
        2., 2.5, 3.,]
    ));

    let src = "
        __kernel void add_vec(__global float* self, __global const float* rhs, __global float* out) {
            size_t id = get_global_id(0);
            out[id] = self[id]+rhs[id];
    }";

    let skip = 1 * lhs.cols();

    let gws = [lhs.size()-skip, 0, 0];
    let out = KernelOptions::<f32>::new(&device, lhs, gws, &src)
        .with_rhs(rhs)
        .with_output(lhs.dims())
        .with_offset([skip, 0, 0])
        .run()?;
    
    //println!("out: {out:?}");
    assert_eq!(out.read(), vec![0., 0., 0., 6., 7.5, 9.]);
    Ok(())
}
*/