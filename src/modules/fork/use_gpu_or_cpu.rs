use core::{hash::BuildHasher, time::Duration};
use std::collections::{BinaryHeap, HashMap};
use std::time::Instant;

use crate::{Analyzation, Fork, GpuOrCpuInfo, HashLocation, UseGpuOrCpu};

pub fn should_use_cpu(
    cpu_op: &mut impl FnMut(),
    gpu_op: &mut impl FnMut(),
) -> (bool, Duration, Duration) {
    let cpu_time_start = Instant::now();
    cpu_op();
    let cpu_time = cpu_time_start.elapsed();

    let gpu_time_start = Instant::now();
    // be sure to sync
    // keep jit compilation overhead in mind
    gpu_op();
    let gpu_time = gpu_time_start.elapsed();

    (cpu_time < gpu_time, cpu_time, gpu_time)
}

pub fn init_binary_heap<S: BuildHasher>(
    cpu_op: &mut impl FnMut(),
    gpu_op: &mut impl FnMut(),
    location: HashLocation<'static>,
    input_lengths: Vec<usize>,
    gpu_or_cpu: &mut HashMap<HashLocation, BinaryHeap<Analyzation>, S>,
) -> GpuOrCpuInfo {
    // removes jit compilation overhead
    // FIXME: algorithm runs twice
    gpu_op();
    let (use_cpu, cpu_dur, gpu_dur) = should_use_cpu(cpu_op, gpu_op);
    // println!("use_cpu: {use_cpu}, cpu_dur: {cpu_dur:?}, gpu_dur: {gpu_dur:?}");
    gpu_or_cpu.insert(
        location,
        BinaryHeap::from([Analyzation {
            input_lengths: input_lengths.to_vec(),
            output_lengths: vec![],
            gpu_dur,
            cpu_dur,
        }]),
    );
    GpuOrCpuInfo {
        use_cpu,
        is_result_cached: false,
    }
}

impl<Mods> UseGpuOrCpu for Fork<Mods> {
    // FIXME: if the operation assigns to  &mut out, you will get chaos
    fn use_cpu_or_gpu(
        &self,
        location: HashLocation<'static>,
        input_lengths: &[usize],
        mut cpu_op: impl FnMut(),
        mut gpu_op: impl FnMut(),
    ) -> GpuOrCpuInfo {

        if !self.enabled.get() {
            gpu_op();
            return GpuOrCpuInfo {
                use_cpu: false,
                is_result_cached: false
            };
        }

        let mut gpu_or_cpu = self.gpu_or_cpu.borrow_mut();

        let Some(operations) = gpu_or_cpu.get_mut(&location) else {
            return init_binary_heap(
                &mut cpu_op,
                &mut gpu_op,
                location,
                input_lengths.to_vec(),
                &mut gpu_or_cpu,
            );
        };

        let anals = operations.clone().into_sorted_vec();

        let input_lengths_sum = input_lengths.iter().sum::<usize>();

        for anals in anals.windows(2) {
            let lhs = &anals[0];
            let rhs = &anals[1];

            let input_lengths_lhs = lhs.input_lengths.iter().sum::<usize>();
            let input_lengths_rhs = rhs.input_lengths.iter().sum::<usize>();

            if input_lengths_sum >= input_lengths_lhs && input_lengths_sum <= input_lengths_rhs {
                let new_cpu_dur = lhs.cpu_dur.as_secs_f32()
                    + 0.5 * (rhs.cpu_dur.as_secs_f32() - lhs.cpu_dur.as_secs_f32());

                let new_gpu_dur = lhs.gpu_dur.as_secs_f32()
                    + 0.5 * (rhs.gpu_dur.as_secs_f32() - lhs.gpu_dur.as_secs_f32());

                let use_cpu = new_cpu_dur < new_gpu_dur;
                match use_cpu {
                    true => cpu_op(),
                    false => gpu_op(),
                }
                return GpuOrCpuInfo {
                    use_cpu,
                    is_result_cached: true,
                };
            }
        }

        gpu_op();
        let (use_cpu, cpu_dur, gpu_dur) = should_use_cpu(&mut cpu_op, &mut gpu_op);
        operations.push(Analyzation {
            input_lengths: input_lengths.to_vec(),
            output_lengths: vec![],
            gpu_dur,
            cpu_dur,
        });

        GpuOrCpuInfo {
            use_cpu,
            is_result_cached: false,
        }
    }

    #[inline]
    fn set_fork_enabled(&self, enabled: bool) {
        self.enabled.set(enabled);
    }

    #[inline] 
    fn is_fork_enabled(&self) -> bool {
        self.enabled.get()
    }
}
