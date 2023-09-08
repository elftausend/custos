use criterion::{criterion_group, criterion_main, Criterion};
use custos::{
    opencl::try_cl_clear, Base, Buffer, Device, Fork, GpuOrCpuInfo, Module, OpenCL, UseGpuOrCpu,
    CPU,
};

pub fn bench_fork(c: &mut Criterion) {

    // only small 
    let sizes = [413, 40, 40, 40, 40, 40, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 432, 120, 431, 204, 100, 430, 42, 451, 631, 321, 420, 320, 140, 350, 20, 40, 50, 60, 30, 30, 30, 30, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 2545213, 121253, 431332, 50000, 431430, 4135410, 1341230, 3124013, 459132, 9513419, 2139412, 3219346,];

    /*let sizes = [
        8_287_587,
        48_941_518,
        29_579_168,
        39_178_476,
        29_450_127,
        123943,
        10031,
        310,
        1230,
        3102,
        31093,
        21934,
        132,
        330,
        30,
        6000,
        2123959,
        212_582_039,
        1349023,
        4923490,
        90,
        8032,
        90_000_000,
    ];*/
    let cpu = CPU::<Base>::new();
    let gpu = OpenCL::<Base>::new(1).unwrap();
    let mut bufs = sizes
        .iter()
        .map(|size| {
            (
                gpu.buffer::<f32, (), _>(vec![1.; *size]),
                cpu.buffer::<f32, (), _>(vec![1.; *size]),
            )
        })
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("fork");

    group.bench_function("cpu-only", |bencher| {
        bencher.iter(|| {
            for (_, cpu_buf) in &mut bufs {
                cpu_buf.clear();
            }
        })
    });

    bufs.clear();

    let mut bufs = sizes
        .iter()
        .map(|size| {
            (
                gpu.buffer::<f32, (), _>(vec![1.; *size]),
                cpu.buffer::<f32, (), _>(vec![1.; *size]),
            )
        })
        .collect::<Vec<_>>();
    group.bench_function("gpu-only", |bencher| {
        bencher.iter(|| {
            for (gpu_buf, _) in &mut bufs {
                try_cl_clear(&gpu, gpu_buf).unwrap();
            }
        })
    });
    let mut bufs = sizes
        .iter()
        .map(|size| {
            (
                gpu.buffer::<f32, (), _>(vec![1.; *size]),
                cpu.buffer::<f32, (), _>(vec![1.; *size]),
            )
        })
        .collect::<Vec<_>>();

    #[track_caller]
    pub fn clear(
        fork: &Fork<Base>,
        cpu_buf: &mut Buffer<f32>,
        opencl_buf: &mut Buffer<f32, OpenCL>,
    ) -> GpuOrCpuInfo {
        fork.use_cpu_or_gpu(
            (file!(), line!(), column!()).into(),
            &[cpu_buf.len()],
            || {
                cpu_buf.clear();
            },
            || opencl_buf.clear(),
        )
    }
    
    let fork = <Fork<Base> as Module<CPU>>::new();
    for (gpu_buf, cpu_buf) in &mut bufs {
        let use_cpu = clear(&fork, cpu_buf, gpu_buf);
        println!("len: {}, use_cpu: {use_cpu:?}", gpu_buf.len());
    }
    group.bench_function("gpu-cpu-fork", |bencher| {
        bencher.iter(|| {
            for (gpu_buf, cpu_buf) in &mut bufs {
                let use_cpu = clear(&fork, cpu_buf, gpu_buf);
                // println!("len: {}, use_cpu: {use_cpu:?}", gpu_buf.len());
            }
        })
    });
}

criterion_group!(benches, bench_fork);
criterion_main!(benches);
