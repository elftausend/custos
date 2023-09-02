use crate::{HashLocation, Module, Setup};
use core::{cell::RefCell, panic::Location, time::Duration};
use std::{
    collections::{BinaryHeap, HashMap},
    time::Instant,
};

pub struct Analyzation {
    input_lengths: Vec<usize>,
    gpu_dur: Duration,
    cpu_dur: Duration,
}

pub struct Fork<Mods> {
    modules: Mods,
    gpu_or_cpu: HashMap<HashLocation<'static>, BinaryHeap<Analyzation>>, // should use Location of operation in file file!(), ...
    use_cpu: RefCell<HashMap<HashLocation<'static>, bool>>, // uses Location::caller() as HashLocation
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

impl<Mods> UseGpuOrCpu for Fork<Mods> {
    // FIXME: if the operation assigns to  &mut out, you will get chaos

    #[inline]
    fn use_cpu_or_gpu(&self, mut cpu_op: impl FnMut(), mut gpu_op: impl FnMut()) -> GpuOrCpu {
        if let Some(use_cpu) = self
            .use_cpu
            .borrow()
            .get(&Location::caller().into())
            .copied()
        {
            match use_cpu {
                true => cpu_op(),
                false => gpu_op(),
            }
            return GpuOrCpu {
                use_cpu,
                is_result_cached: true,
            };
        }

        let cpu_time_start = Instant::now();
        cpu_op();
        let cpu_time = cpu_time_start.elapsed();

        let gpu_time_start = Instant::now();
        // be sure to sync
        // keep jit compilation overhead in mind
        gpu_op();
        let gpu_time = gpu_time_start.elapsed();

        let use_cpu = cpu_time < gpu_time;

        self.use_cpu
            .borrow_mut()
            .insert(Location::caller().into(), use_cpu);

        GpuOrCpu {
            use_cpu,
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
    fn use_cpu_or_gpu(&self, cpu_op: impl FnMut(), gpu_op: impl FnMut()) -> GpuOrCpu;
}

impl<Mods: UseGpuOrCpu> UseGpuOrCpu for crate::OpenCL<Mods> {
    fn use_cpu_or_gpu(&self, cpu_op: impl FnMut(), gpu_op: impl FnMut()) -> GpuOrCpu {
        let gpu_or_cpu = self.modules.use_cpu_or_gpu(cpu_op, gpu_op);
        if !gpu_or_cpu.is_result_cached {
            return gpu_or_cpu;
        }

        // gpu_op();

        gpu_or_cpu
    }
}

#[cfg(test)]
mod tests {
    use crate::{Base, Buffer, Device, Fork, GpuOrCpu, Module, OpenCL, UseGpuOrCpu, CPU};

    #[track_caller]
    pub fn clear(
        fork: &Fork<Base>,
        cpu_buf: &mut Buffer<i32>,
        opencl_buf: &mut Buffer<i32, OpenCL>,
    ) -> GpuOrCpu {
        fork.use_cpu_or_gpu(
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
        opencl_buf.clear();

        for _ in 0..10 {
            let use_cpu = clear(&fork, &mut cpu_buf, &mut opencl_buf);
            println!("use_cpu: {use_cpu:?}")
        }
    }

    #[test]
    fn test_fork_module() {
        let device = OpenCL::<Fork<Base>>::new(0).unwrap();
    }
}
