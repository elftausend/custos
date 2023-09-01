use core::{time::Duration, cell::RefCell, panic::Location};
use std::{collections::{HashMap, BinaryHeap}, time::Instant};
use crate::{HashLocation, Module};

pub struct Analyzation {
    input_lengths: Vec<usize>,
    gpu_dur: Duration,
    cpu_dur: Duration
}

pub struct Fork<Mods> {
    modules: Mods,
    gpu_or_cpu: HashMap<HashLocation<'static>, BinaryHeap<Analyzation>>, // should use Location of operation in file file!(), ...
    use_cpu: RefCell<HashMap<HashLocation<'static>, bool>> // uses Location::caller() as HashLocation
    
}

impl<Mods: Module<D>, D> Module<D> for Fork<Mods> {
    type Module = Fork<Mods::Module>;

    #[inline]
    fn new() -> Self::Module {
        Fork {
            modules: Mods::new(),
            gpu_or_cpu: Default::default(),
            use_cpu: Default::default() 
        }
    }
}

impl<Mods> UseGpuOrCpu for Fork<Mods> {
    #[inline]
    fn use_cpu_or_gpu(&self, cpu_op: impl Fn(), gpu_op: impl Fn()) -> bool {
        if let Some(use_cpu) = self.use_cpu.borrow().get(&Location::caller().into()) {
            return *use_cpu;
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

        self.use_cpu.borrow_mut().insert(Location::caller().into(), use_cpu);

        use_cpu
    }
}

pub trait UseGpuOrCpu {
    #[track_caller]
    fn use_cpu_or_gpu(&self, cpu_op: impl Fn(), gpu_op: impl Fn()) -> bool;
}
