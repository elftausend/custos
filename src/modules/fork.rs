use crate::{
    Device, HashLocation, LocationHasher, Module, OnDropBuffer, OnNewBuffer, Setup, Shape, OpenCL, Base, CPU,
};
use core::{cell::RefCell, hash::BuildHasherDefault, panic::Location, time::Duration, borrow::BorrowMut};
use std::{
    collections::{BinaryHeap, HashMap},
    time::Instant,
};

#[derive(Debug, PartialEq, Eq, PartialOrd)]
pub struct Analyzation {
    input_lengths: Vec<usize>,
    output_lengths: Vec<usize>,
    gpu_dur: Duration,
    cpu_dur: Duration,
}

impl Ord for Analyzation {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.cpu_dur.cmp(&other.cpu_dur)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ForkNode {
    use_cpu: bool,
    backoff: usize,
    retrieves: usize,
}

pub struct Fork<Mods> {
    modules: Mods,
    gpu_or_cpu: RefCell<HashMap<HashLocation<'static>, BinaryHeap<Analyzation>>>, // should use Location of operation in file file!(), ...
    use_cpu: RefCell<HashMap<HashLocation<'static>, ForkNode, BuildHasherDefault<LocationHasher>>>, // uses Location::caller() as HashLocation
}

impl<Mods: Module<D>, D> Module<D> for Fork<Mods> {
    type Module = Fork<Mods::Module>;

    #[inline]
    fn new() -> Self::Module {
        Fork {
            modules: Mods::new(),
            gpu_or_cpu: Default::default(),
            use_cpu: Default::default(),
        }
    }
}
pub trait ForkSetup {
    #[inline]
    fn fork_setup(&mut self) {}
}

impl<Mods: Setup<D>, D: ForkSetup> Setup<D> for Fork<Mods> {
    fn setup(device: &mut D) {
        // check if device supports unified memory
        device.fork_setup();
        Mods::setup(device)
    }
}

pub fn should_use_cpu(mut cpu_op: impl FnMut(), mut gpu_op: impl FnMut()) -> bool {
    let cpu_time_start = Instant::now();
    cpu_op();
    let cpu_time = cpu_time_start.elapsed();

    let gpu_time_start = Instant::now();
    // be sure to sync
    // keep jit compilation overhead in mind
    gpu_op();
    let gpu_time = gpu_time_start.elapsed();

    cpu_time < gpu_time
}

impl<Mods> UseGpuOrCpu for Fork<Mods> {
    // FIXME: if the operation assigns to  &mut out, you will get chaos
    fn use_cpu_or_gpu(&self, location: HashLocation<'static>, input_lengths: &[usize], mut cpu_op: impl FnMut(), mut gpu_op: impl FnMut()) -> GpuOrCpu {

        GpuOrCpu {
            use_cpu: false,
            is_result_cached: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct GpuOrCpu {
    pub use_cpu: bool,
    pub is_result_cached: bool,
}

pub trait UseGpuOrCpu {
    #[track_caller]
    fn use_cpu_or_gpu(&self, location: HashLocation<'static>, input_lengths: &[usize], cpu_op: impl FnMut(), gpu_op: impl FnMut()) -> GpuOrCpu;
}

impl<Mods: UseGpuOrCpu> UseGpuOrCpu for crate::OpenCL<Mods> {
    fn use_cpu_or_gpu(&self, location: HashLocation<'static>, input_lengths: &[usize], cpu_op: impl FnMut(), gpu_op: impl FnMut()) -> GpuOrCpu {
        let gpu_or_cpu = self.modules.use_cpu_or_gpu(location, input_lengths, cpu_op, gpu_op);
        if !gpu_or_cpu.is_result_cached {
            return gpu_or_cpu;
        }

        // gpu_op();

        gpu_or_cpu
    }
}

impl<Mods: OnDropBuffer> OnDropBuffer for Fork<Mods> {
    #[inline]
    fn on_drop_buffer<T, D: crate::Device, S: crate::Shape>(
        &self,
        device: &D,
        buf: &crate::Buffer<T, D, S>,
    ) {
        self.modules.on_drop_buffer(device, buf)
    }
}

impl<Mods: OnNewBuffer<T, D, S>, T, D: Device, S: Shape> OnNewBuffer<T, D, S> for Fork<Mods> {
    #[inline]
    fn on_new_buffer(&self, device: &D, new_buf: &crate::Buffer<T, D, S>) {
        self.modules.on_new_buffer(device, new_buf)
    }
}

pub(crate) fn measure_kernel_overhead_opencl<Mods>(device: &crate::OpenCL<Mods>) {
    let src = "
        __kernel void measureJit() {
            
        }
    ";
    device.launch_kernel(src, [1, 0, 0], None, &[]).unwrap();
}

#[test]
fn test_linear() {
    let fork = <Fork<Base> as Module<CPU>>::new();
    let mut heap = BinaryHeap::new();
    let anals = [
        Analyzation {
            input_lengths: vec![100000, 100000],
            output_lengths: vec![100000],
            gpu_dur: std::time::Duration::from_secs_f32(0.312),
            cpu_dur: std::time::Duration::from_secs_f32(0.12),
        },
        Analyzation {
            input_lengths: vec![140000, 140000],
            output_lengths: vec![140000],
            gpu_dur: std::time::Duration::from_secs_f32(0.412),
            cpu_dur: std::time::Duration::from_secs_f32(0.52),
        }
    ];

    let input_lengths = vec![120000, 120000];
    let output_lengths = vec![120000];

    for anal in anals {
        heap.push(anal)
    }
    let input_lengths = input_lengths.iter().sum::<usize>();
    let output_lengths = output_lengths.iter().sum::<usize>();
    
    let anals = heap.into_sorted_vec();

    for anals in anals.windows(2) {
        let lhs = &anals[0];
        let rhs = &anals[1];

        let input_lengths_lhs = lhs.input_lengths.iter().sum::<usize>();
        let output_lengths_lhs = lhs.output_lengths.iter().sum::<usize>();
        
        let input_lengths_rhs = rhs.input_lengths.iter().sum::<usize>();
        let output_lengths_rhs = rhs.output_lengths.iter().sum::<usize>();

        if input_lengths > input_lengths_lhs && input_lengths < input_lengths_rhs {
            let new_cpu_dur = lhs.cpu_dur + ((rhs.cpu_dur - lhs.cpu_dur) / 2);
            println!("new_cpu_dur: {new_cpu_dur:?}");

            let new_gpu_dur = lhs.gpu_dur + ((rhs.gpu_dur - lhs.gpu_dur) / 2);

            println!("new_gpu_dur: {new_gpu_dur:?}");
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        opencl::try_cl_clear, Base, Buffer, Device, Fork, GpuOrCpu, Module, OpenCL, UseGpuOrCpu,
        CPU,
    };

    #[track_caller]
    pub fn clear(
        fork: &Fork<Base>,
        cpu_buf: &mut Buffer<i32>,
        opencl_buf: &mut Buffer<i32, OpenCL>,
    ) -> GpuOrCpu {
        fork.use_cpu_or_gpu((file!(), line!(), column!()).into(), &[],
            || {
                cpu_buf.clear();
            },
            || opencl_buf.clear(),
        )
    }

    #[test]
    fn test_use_gpu_or_cpu() {
        let fork = <Fork<Base> as Module<CPU>>::new();
        const SIZE: usize = 100000000;
        let device = CPU::<Base>::new();
        let mut cpu_buf = device.buffer::<_, (), _>(vec![1; SIZE]);

        let opencl = OpenCL::<Base>::new(0).unwrap();
        let mut opencl_buf = opencl.buffer::<_, (), _>(vec![1; SIZE]);
        // opencl_buf.clear();

        for _ in 0..100 {
            let use_cpu = clear(&fork, &mut cpu_buf, &mut opencl_buf);
            println!("use_cpu: {use_cpu:?}")
        }
    }

    #[test]
    fn test_fork_module() {
        let device = OpenCL::<Fork<Base>>::new(0).unwrap();

        let mut buf = device.buffer::<_, (), _>(vec![21u8; 10000000]);

        // this is for the jit warming
        try_cl_clear(&device, &mut buf).unwrap();

        buf.clear();
    }
}
