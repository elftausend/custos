use crate::{
    impl_remove_layer, pass_down_add_operation, pass_down_exec_now, pass_down_tape_actions,
    AddLayer, Alloc, Buffer, Device, HasId, HashLocation, IsShapeIndep, LocationHasher, Module,
    OnDropBuffer, OnNewBuffer, Parents, PtrType, Retrieve, RunModule, Setup, Shape, WrappedData,
};
use core::{cell::RefCell, hash::BuildHasherDefault};
use std::collections::{BinaryHeap, HashMap};

mod analyzation;
mod fork_macro;
mod use_gpu_or_cpu;

pub use analyzation::Analyzation;
pub use use_gpu_or_cpu::*;

pub struct Fork<Mods> {
    pub modules: Mods,
    pub gpu_or_cpu: RefCell<
        HashMap<HashLocation<'static>, BinaryHeap<Analyzation>, BuildHasherDefault<LocationHasher>>,
    >, // should use Location of operation in file file!(), ...
}

impl<Mods: WrappedData> WrappedData for Fork<Mods> {
    type Wrap<T, Base: HasId + PtrType> = Mods::Wrap<T, Base>;

    #[inline]
    fn wrap_in_base<T, Base: HasId + PtrType>(&self, base: Base) -> Self::Wrap<T, Base> {
        self.modules.wrap_in_base(base)
    }

    #[inline]
    fn wrapped_as_base<'a, T, Base: HasId + PtrType>(wrap: &'a Self::Wrap<T, Base>) -> &'a Base {
        Mods::wrapped_as_base(wrap)
    }

    #[inline]
    fn wrapped_as_base_mut<'a, T, Base: HasId + PtrType>(
        wrap: &'a mut Self::Wrap<T, Base>,
    ) -> &'a mut Base {
        Mods::wrapped_as_base_mut(wrap)
    }
}

impl<Mods: Module<D>, D: Device> Module<D> for Fork<Mods> {
    type Module = Fork<Mods::Module>;

    #[inline]
    fn new() -> Self::Module {
        Fork {
            modules: Mods::new(),
            gpu_or_cpu: Default::default(),
        }
    }
}
pub trait ForkSetup {
    #[inline]
    fn fork_setup(&mut self) {}
}

impl<Mods: Setup<D>, D: ForkSetup> Setup<D> for Fork<Mods> {
    fn setup(device: &mut D) -> crate::Result<()> {
        // check if device supports unified memory
        device.fork_setup();
        Mods::setup(device)
    }
}

pass_down_add_operation!(Fork);
pass_down_exec_now!(Fork);

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

impl<T: 'static, Mods: Retrieve<D, T, S>, D: IsShapeIndep + 'static, S: Shape> Retrieve<D, T, S>
    for Fork<Mods>
{
    #[inline]
    fn retrieve<const NUM_PARENTS: usize>(
        &self,
        device: &D,
        len: usize,
        parents: impl Parents<NUM_PARENTS>,
    ) -> Self::Wrap<T, D::Base<T, S>>
    where
        S: Shape,
        D: Alloc<T>,
    {
        self.modules.retrieve(device, len, parents)
    }

    #[inline]
    fn on_retrieve_finish(&self, retrieved_buf: &Buffer<T, D, S>)
    where
        D: Alloc<T>,
    {
        // pass down
        self.modules.on_retrieve_finish(retrieved_buf)
    }
}

pass_down_tape_actions!(Fork);

impl<Mods: RunModule<D>, D> RunModule<D> for Fork<Mods> {
    #[inline]
    fn run(&self, _device: &D) -> crate::Result<()> {
        self.modules.run(_device)
    }
}

impl_remove_layer!(Fork);

impl<NewMods, SD> AddLayer<NewMods, SD> for Fork<()> {
    type Wrapped = crate::Fork<NewMods>;

    #[inline]
    fn wrap_layer(inner_mods: NewMods) -> Self::Wrapped {
        Fork {
            modules: inner_mods,
            gpu_or_cpu: Default::default(),
        }
    }
}

#[cfg(test)]
#[cfg(feature = "opencl")]
mod tests {
    use std::{collections::BinaryHeap, time::Instant};

    use crate::{
        opencl::try_cl_clear, should_use_cpu, Analyzation, ApplyFunction, Base, Buffer, Cached,
        Combiner, Device, Fork, GpuOrCpuInfo, Module, OpenCL, UseGpuOrCpu, CPU,
    };

    #[track_caller]
    pub fn clear(
        fork: &Fork<Base>,
        cpu_buf: &mut Buffer<i32>,
        opencl_buf: &mut Buffer<i32, OpenCL>,
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

    #[test]
    #[ignore]
    fn test_use_gpu_or_cpu() {
        let fork = <Fork<Base> as Module<CPU>>::new();
        // const SIZE: usize = 100000000;
        const SIZE: usize = 41_518;
        let device = CPU::<Base>::new();
        let mut cpu_buf = device.buffer::<_, (), _>(vec![1; SIZE]);

        let opencl = OpenCL::<Base>::new(1).unwrap();
        let mut opencl_buf = opencl.buffer::<_, (), _>(vec![1; SIZE]);
        // opencl_buf.clear();

        for _ in 0..100 {
            let use_cpu = clear(&fork, &mut cpu_buf, &mut opencl_buf);
            println!("use_cpu: {use_cpu:?}")
        }
    }

    #[test]
    #[ignore]
    fn test_diff_sizes() {
        let cpu = CPU::<Base>::new();
        let gpu = OpenCL::<Base>::new(1).unwrap();

        let sizes = [8_287_587, 48_941_518];

        let mut bufs = sizes
            .iter()
            .map(|size| {
                (
                    gpu.buffer::<i32, (), _>(vec![1; *size]),
                    cpu.buffer::<i32, (), _>(vec![1; *size]),
                )
            })
            .collect::<Vec<_>>();
        // for (gpu_buf, cpu_buf) in &mut bufs {
        // gpu_buf.clear();
        // }

        let fork = <Fork<Base> as Module<CPU>>::new();
        for (gpu_buf, cpu_buf) in &mut bufs {
            let res = should_use_cpu(&mut || cpu_buf.clear(), &mut || gpu_buf.clear());
            println!("res: {res:?}");

            let res = clear(&fork, cpu_buf, gpu_buf);
            println!("gpu_or_cpu_info: {res:?}");
            cpu_buf.clear();
            let start = Instant::now();
            gpu_buf.clear();
            let elapsed = start.elapsed();
            println!("elapsed: {elapsed:?}")
        }
    }

    #[test]
    #[ignore]
    fn test_check_for_reasonable_fork_execution_time() {
        let cpu = CPU::<Base>::new();
        let gpu = OpenCL::<Base>::new(1).unwrap();

        let sizes = [
            8_287_587,
            48_941_518,
            /*59_579_168,
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
            3123959,
            312_582_039,
            1349023,
            4923490,
            90,
            8032,
            100_000_000,*/
        ];

        let mut bufs = sizes
            .iter()
            .map(|size| {
                (
                    gpu.buffer::<i32, (), _>(vec![1; *size]),
                    cpu.buffer::<i32, (), _>(vec![1; *size]),
                )
            })
            .collect::<Vec<_>>();

        let fork = <Fork<Base> as Module<CPU>>::new();
        for (gpu_buf, cpu_buf) in &mut bufs {
            let use_cpu = clear(&fork, cpu_buf, gpu_buf);
            println!("len: {}, use_cpu: {use_cpu:?}", gpu_buf.len());
        }
    }

    #[test]
    #[ignore]
    fn test_use_gpu_or_cpu_varying_sizes() {
        let fork = <Fork<Base> as Module<CPU>>::new();

        let sizes = [
            8287587,
            48_941_518,
            59_579_168,
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
            3123959,
            312_582_039,
            1349023,
            4923490,
            90,
            8032,
            100_000_000,
        ];

        let device = CPU::<Base>::new();

        let opencl = OpenCL::<Base>::new(1).unwrap();
        println!("name: {:?}", opencl.name());

        // opencl_buf.clear();

        for _ in 0..1000 {
            for size in sizes {
                let mut cpu_buf = device.buffer::<_, (), _>(vec![1; size]);

                let mut opencl_buf = opencl.buffer::<_, (), _>(vec![1; size]);
                let use_cpu = clear(&fork, &mut cpu_buf, &mut opencl_buf);
                println!("use_cpu: {use_cpu:?}")
            }
        }
        println!("{:?}", fork.gpu_or_cpu.borrow());
    }

    #[test]
    fn test_fork_module() {
        let device = OpenCL::<Fork<Base>>::new(0).unwrap();

        let mut buf = device.buffer::<_, (), _>(vec![21u8; 10000]);

        // this is for the jit warming
        try_cl_clear(&device, &mut buf).unwrap();

        buf.clear();
    }
    #[test]
    fn test_lerp_of_analyzation_time() {
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
            },
        ];

        let input_lengths = [120000, 120000];
        // let output_lengths = vec![120000];

        for anal in anals {
            heap.push(anal)
        }
        let input_lengths = input_lengths.iter().sum::<usize>();

        let anals = heap.into_sorted_vec();

        for anals in anals.windows(2) {
            let lhs = &anals[0];
            let rhs = &anals[1];

            let input_lengths_lhs = lhs.input_lengths.iter().sum::<usize>();

            let input_lengths_rhs = rhs.input_lengths.iter().sum::<usize>();

            if input_lengths > input_lengths_lhs && input_lengths < input_lengths_rhs {
                let new_cpu_dur = lhs.cpu_dur + ((rhs.cpu_dur - lhs.cpu_dur) / 2);
                let new_gpu_dur = lhs.gpu_dur + ((rhs.gpu_dur - lhs.gpu_dur) / 2);

                assert_eq!(new_cpu_dur.as_secs_f32(), 0.32);

                assert_eq!(new_gpu_dur.as_secs_f32(), (0.312 + (0.412 - 0.312) * 0.5));
            }
        }
    }

    #[cfg(unified_cl)]
    #[test]
    fn test_fork_with_opencl_device_and_apply_fn() {
        let device = OpenCL::<Fork<Cached<Base>>>::new(0).unwrap();
        if !device.unified_mem() {
            return;
        }
        let buf = device.buffer([1, 2, 4, 5, 6, 7]);
        let out = device.apply_fn(&buf, |x| x.add(3));
        assert_eq!(out.read(), [4, 5, 7, 8, 9, 10]);

        for _ in 0..100 {
            let _out = device.apply_fn(&buf, |x| x.add(3));
            let gpu_or_cpu = device.modules.gpu_or_cpu.borrow();
            let (_, operations) = gpu_or_cpu.iter().next().unwrap();
            assert_eq!(operations.len(), 2);
            let analyzations = operations.iter().cloned().collect::<Vec<Analyzation>>();
            assert_eq!(&analyzations[0].input_lengths, &[6]);
        }
    }
}
