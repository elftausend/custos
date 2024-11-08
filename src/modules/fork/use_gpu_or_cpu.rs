use core::{hash::BuildHasher, time::Duration};
use std::collections::{BinaryHeap, HashMap};
use std::time::Instant;

use crate::analyzation::Analyzation;

use crate::{Fork, GpuOrCpuInfo, HashLocation, UseGpuOrCpu};

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
    // Removes jit compilation overhead
    gpu_op();
    let (use_cpu, cpu_dur, gpu_dur) = should_use_cpu(cpu_op, gpu_op);
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

pub fn extrapolate_and_execute(
    lhs: &Analyzation,
    rhs: &Analyzation,
    input_lengths_sum: usize,
    mut cpu_op: impl FnMut(),
    mut gpu_op: impl FnMut(),
) -> GpuOrCpuInfo {
    // Perform linear extrapolation using the trend between lhs and rhs
    let input_lengths_lhs = lhs.input_lengths.iter().sum::<usize>();
    let input_lengths_rhs = rhs.input_lengths.iter().sum::<usize>();

    let cpu_slope = (rhs.cpu_dur.as_secs_f32() - lhs.cpu_dur.as_secs_f32())
        / (input_lengths_rhs - input_lengths_lhs) as f32;

    let gpu_slope = (rhs.gpu_dur.as_secs_f32() - lhs.gpu_dur.as_secs_f32())
        / (input_lengths_rhs - input_lengths_lhs) as f32;

    // Extrapolate durations based on input length
    let new_cpu_dur =
        lhs.cpu_dur.as_secs_f32() + cpu_slope * (input_lengths_sum - input_lengths_lhs) as f32;
    let new_gpu_dur =
        lhs.gpu_dur.as_secs_f32() + gpu_slope * (input_lengths_sum - input_lengths_lhs) as f32;

    let use_cpu = new_cpu_dur < new_gpu_dur;
    match use_cpu {
        true => cpu_op(),
        false => gpu_op(),
    }

    GpuOrCpuInfo {
        use_cpu,
        is_result_cached: true,
    }
}

impl<Mods> UseGpuOrCpu for Fork<Mods> {
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
                is_result_cached: false,
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

        // Extrapolate if input is outside known range
        if input_lengths_sum < anals.first().unwrap().input_lengths.iter().sum::<usize>() {
            let lhs = &anals[0];
            let rhs = &anals[1];
            return self.extrapolate_and_execute(lhs, rhs, input_lengths_sum, cpu_op, gpu_op);
        } else if input_lengths_sum > anals.last().unwrap().input_lengths.iter().sum::<usize>() {
            let lhs = &anals[anals.len() - 2];
            let rhs = &anals[anals.len() - 1];
            return self.extrapolate_and_execute(lhs, rhs, input_lengths_sum, cpu_op, gpu_op);
        }

        // If within range, perform interpolation
        for anals in anals.windows(2) {
            let lhs = &anals[0];
            let rhs = &anals[1];

            let input_lengths_lhs = lhs.input_lengths.iter().sum::<usize>();
            let input_lengths_rhs = rhs.input_lengths.iter().sum::<usize>();

            if input_lengths_sum >= input_lengths_lhs && input_lengths_sum <= input_lengths_rhs {
                let new_cpu_dur = lhs.cpu_dur.as_secs_f32()
                    + (input_lengths_sum - input_lengths_lhs) as f32
                        / (input_lengths_rhs - input_lengths_lhs) as f32
                        * (rhs.cpu_dur.as_secs_f32() - lhs.cpu_dur.as_secs_f32());

                let new_gpu_dur = lhs.gpu_dur.as_secs_f32()
                    + (input_lengths_sum - input_lengths_lhs) as f32
                        / (input_lengths_rhs - input_lengths_lhs) as f32
                        * (rhs.gpu_dur.as_secs_f32() - lhs.gpu_dur.as_secs_f32());

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

        // Perform actual measurement and caching if no interpolation or extrapolation matched
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn mock_cpu_op() {
        // Simulate CPU operation (this can be left empty)
    }

    fn mock_gpu_op() {
        // Simulate GPU operation (this can be left empty)
    }

    #[test]
    fn test_extrapolate_and_execute_cpu_faster() {
        let lhs = Analyzation {
            input_lengths: vec![100],
            output_lengths: vec![],
            cpu_dur: Duration::from_secs_f32(1.0), // CPU took 1 sec
            gpu_dur: Duration::from_secs_f32(2.0), // GPU took 2 secs
        };

        let rhs = Analyzation {
            input_lengths: vec![200],
            output_lengths: vec![],
            cpu_dur: Duration::from_secs_f32(1.5), // CPU took 1.5 secs
            gpu_dur: Duration::from_secs_f32(2.5), // GPU took 2.5 secs
        };

        let input_lengths_sum = 150;

        let result =
            extrapolate_and_execute(&lhs, &rhs, input_lengths_sum, mock_cpu_op, mock_gpu_op);

        assert!(
            result.use_cpu,
            "CPU should be faster based on extrapolation."
        );
        assert!(
            result.is_result_cached,
            "Result should be cached after extrapolation."
        );
    }

    #[test]
    fn test_extrapolate_and_execute_gpu_faster() {
        let lhs = Analyzation {
            input_lengths: vec![100],
            output_lengths: vec![],
            cpu_dur: Duration::from_secs_f32(2.0), // CPU took 2 secs
            gpu_dur: Duration::from_secs_f32(1.0), // GPU took 1 sec
        };

        let rhs = Analyzation {
            input_lengths: vec![200],
            output_lengths: vec![],
            cpu_dur: Duration::from_secs_f32(2.5), // CPU took 2.5 secs
            gpu_dur: Duration::from_secs_f32(1.5), // GPU took 1.5 secs
        };

        let input_lengths_sum = 150;

        let result =
            extrapolate_and_execute(&lhs, &rhs, input_lengths_sum, mock_cpu_op, mock_gpu_op);

        assert!(
            !result.use_cpu,
            "GPU should be faster based on extrapolation."
        );
        assert!(
            result.is_result_cached,
            "Result should be cached after extrapolation."
        );
    }
}
