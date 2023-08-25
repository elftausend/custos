use std::time::Instant;

use custos::{
    prelude::Number,
    range,
    wgpu::{launch_shader, WGPU},
    Buffer, Device, OpenCL, Shape,
};

use super::ElementWise;

pub fn wgpu_element_wise<T: Number, S: Shape>(
    device: &WGPU,
    lhs: &Buffer<T, WGPU, S>,
    rhs: &Buffer<T, WGPU, S>,
    out: &mut Buffer<T, WGPU, S>,
    op: &str,
) {
    let src = format!(
        "@group(0)
        @binding(0)
        var<storage, read_write> a: array<{datatype}>;
        
        @group(0)
        @binding(1)
        var<storage, read_write> b: array<{datatype}>;

        @group(0)
        @binding(2)
        var<storage, read_write> out: array<{datatype}>;
        
        
        @compute
        @workgroup_size(8, 1, 1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            out[global_id.x] = a[global_id.x] {op} b[global_id.x];
        }}
        ",
        datatype = std::any::type_name::<T>()
    );

    launch_shader(device, &src, [lhs.len() as u32, 1, 1], &[lhs, rhs, out]);
}

impl<T: Number, S: Shape> ElementWise<T, WGPU, S> for WGPU {
    #[inline]
    fn add(&self, lhs: &Buffer<T, WGPU, S>, rhs: &Buffer<T, WGPU, S>) -> Buffer<T, WGPU, S> {
        let mut out = self.retrieve(lhs.len(), (lhs, rhs));
        wgpu_element_wise(self, lhs, rhs, &mut out, "+");
        out
    }

    #[inline]
    fn mul(&self, lhs: &Buffer<T, WGPU, S>, rhs: &Buffer<T, WGPU, S>) -> Buffer<T, WGPU, S> {
        let mut out = self.retrieve(lhs.len(), (lhs, rhs));
        wgpu_element_wise(self, lhs, rhs, &mut out, "*");
        out
    }
}

#[test]
fn test_add() {
    let device = WGPU::new(wgpu::Backends::all()).unwrap();
    let lhs = Buffer::<f32, _>::from((&device, &[1., 2., 3., 4., -9.]));
    let rhs = Buffer::<f32, _>::from((&device, &[1., 2., 3., 4., -9.]));

    for _ in 0..1 {
        let out = device.add(&lhs, &rhs);
    }

    //   println!("read: {:?}", out.read());
}

#[test]
fn test_add_large() {
    const N: usize = 65535;

    let rhs_data = (0..N)
        .into_iter()
        .map(|val| val as f32)
        .collect::<Vec<f32>>();
    let out_actual_data = (0..N)
        .into_iter()
        .map(|val| val as f32 + 1.)
        .collect::<Vec<f32>>();

    let device = WGPU::new(wgpu::Backends::all()).unwrap();

    let lhs = Buffer::<f32, _>::from((&device, &[1.; N]));
    let rhs = Buffer::<f32, _>::from((&device, &rhs_data));

    let start = Instant::now();

    for _ in range(0..100) {
        let out = device.add(&lhs, &rhs);
        assert_eq!(out.read(), out_actual_data);
    }

    println!("wgpu dur: {:?}", start.elapsed());

    let device = OpenCL::<Base>::new(chosen_cl_idx()).unwrap();

    let lhs = Buffer::<f32, _>::from((&device, &[1.; N]));
    let rhs = Buffer::<f32, _>::from((&device, &rhs_data));

    let start = Instant::now();
    for _ in range(0..100) {
        let out = device.add(&lhs, &rhs);
        assert_eq!(out.read(), out_actual_data);
    }

    println!("ocl dur: {:?}", start.elapsed());

    //   println!("read: {:?}", out.read());
}
