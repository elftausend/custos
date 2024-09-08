use core::mem::ManuallyDrop;

use crate::CUDA;

use super::api::{
    create_graph_execution, create_graph_from_captured_stream, cuGraphLaunch, cuStreamBeginCapture,
    CUStreamCaptureMode, CudaErrorKind, Graph, GraphExec, Stream,
};

pub struct LazyCudaGraph {
    pub graph: Graph,
    // ManuallyDrop: cuGraphExecDestroy does not exit??
    graph_exec: ManuallyDrop<GraphExec>,
}

impl LazyCudaGraph {
    pub fn new(stream: &Stream) -> Result<Self, CudaErrorKind> {
        let graph = create_graph_from_captured_stream(stream)?;
        let graph_exec = ManuallyDrop::new(create_graph_execution(&graph)?);

        Ok(LazyCudaGraph { graph, graph_exec })
    }

    pub fn launch(&self, stream: &Stream) -> Result<(), CudaErrorKind> {
        unsafe { cuGraphLaunch(self.graph_exec.0.as_ptr(), stream.0).to_result()? }
        Ok(())
    }
}

#[cfg(feature = "lazy")]
impl<Mods> crate::LazyRun for CUDA<Mods> {
    #[inline]
    fn run(&self) -> crate::Result<()> {
        let graph = self
            .graph
            // TODO: change to get_or_try_init when stable
            // an error may occur if the stream was empty ig
            .get_or_init(|| LazyCudaGraph::new(self.stream()));

        match graph {
            Ok(graph) => {
                graph.launch(&self.stream)?;
                self.stream().sync()?;
            }
            Err(e) => return Err((*e).into()),
        }
        Ok(())
    }
}

impl<Mods: crate::RunModule<Self>> crate::Run for CUDA<Mods> {
    #[inline]
    unsafe fn run(&self) -> crate::Result<()> {
        self.modules.run(self)
    }
}

#[cfg(feature = "lazy")]
impl<Mods> crate::LazySetup for CUDA<Mods> {
    #[inline]
    fn lazy_setup(&mut self) -> crate::Result<()> {
        // switch to stream record mode for graph
        unsafe {
            cuStreamBeginCapture(
                self.stream.0,
                CUStreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
            )
        }
        .to_result()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        AddOperation, ApplyFunction, Base, Buffer, Combiner, Device, HasId, Lazy, Retrieve,
        Retriever, Run, CUDA,
    };

    pub fn ew_src(fn_name: &str, operator: char) -> String {
        format!(
            r#"
            extern "C" __global__ void {fn_name}(int* lhs, int* rhs, int* out, int len) {{
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= len) {{
                    return;
                }}
                out[idx] = lhs[idx] {operator} rhs[idx];
            }}
        "#
        )
    }

    #[test]
    fn test_lazy_cuda_run() {
        let device = CUDA::<Lazy<Base>>::new(0).unwrap();
        let mut lhs = device.buffer([1, 2, 3, 4, 5, 6]);
        let mut rhs = device.buffer([1, 2, 3, 4, 5, 6]);
        let mut out = lhs.empty_like();

        let add_src = ew_src("add", '+');
        let mul_src = ew_src("mul", '*');

        device
            .launch_kernel1d(
                lhs.len(),
                &add_src,
                "add",
                &[&lhs, &rhs, &mut out, &lhs.len()],
            )
            .unwrap();

        device
            .launch_kernel1d(
                lhs.len(),
                &add_src,
                "add",
                &[&out, &lhs, &mut rhs, &lhs.len()],
            )
            .unwrap();

        device
            .launch_kernel1d(
                rhs.len(),
                &mul_src,
                "mul",
                &[&out, &rhs, &mut lhs, &rhs.len()],
            )
            .unwrap();

        assert_eq!(out.read(), vec![0; out.len()]);
        assert_eq!(lhs.read(), vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(rhs.read(), vec![1, 2, 3, 4, 5, 6]);

        unsafe { device.run().unwrap() };

        assert_eq!(out.read(), vec![2, 4, 6, 8, 10, 12]);
        assert_eq!(rhs.read(), vec![3, 6, 9, 12, 15, 18]);
        assert_eq!(lhs.read(), vec![6, 24, 54, 96, 150, 216]);
    }

    #[test]
    fn test_lazy_cuda_run_multiple_times() {
        let device = CUDA::<Lazy<Base>>::new(0).unwrap();
        let mut lhs = device.buffer([1, 2, 3, 4, 5, 6]);
        let mut rhs = device.buffer([1, 2, 3, 4, 5, 6]);
        let mut out = lhs.empty_like();

        let add_src = ew_src("add", '+');
        let mul_src = ew_src("mul", '*');

        out.clear();

        device
            .launch_kernel1d(
                lhs.len(),
                &add_src,
                "add",
                &[&lhs, &rhs, &mut out, &lhs.len()],
            )
            .unwrap();

        device
            .launch_kernel1d(
                lhs.len(),
                &add_src,
                "add",
                &[&out, &lhs, &mut rhs, &lhs.len()],
            )
            .unwrap();

        device
            .launch_kernel1d(
                rhs.len(),
                &mul_src,
                "mul",
                &[&out, &rhs, &mut lhs, &rhs.len()],
            )
            .unwrap();

        assert_eq!(out.read(), vec![0; out.len()]);
        assert_eq!(lhs.read(), vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(rhs.read(), vec![1, 2, 3, 4, 5, 6]);

        for _ in 0..10 {
            lhs.write(&[1, 2, 3, 4, 5, 6]);
            rhs.write(&[1, 2, 3, 4, 5, 6]);
            device.mem_transfer_stream.sync().unwrap();
            unsafe { device.run().unwrap() };
        }

        assert_eq!(out.read(), vec![2, 4, 6, 8, 10, 12]);
        assert_eq!(rhs.read(), vec![3, 6, 9, 12, 15, 18]);
        assert_eq!(lhs.read(), vec![6, 24, 54, 96, 150, 216]);
    }

    #[test]
    fn test_cuda_eager_without_lazy() {
        let device = CUDA::<Base>::new(0).unwrap();
        let mut lhs = device.buffer([1, 2, 3, 4, 5, 6]);
        let mut rhs = device.buffer([1, 2, 3, 4, 5, 6]);
        let mut out = lhs.empty_like();

        let add_src = ew_src("add", '+');
        let mul_src = ew_src("mul", '*');

        assert_eq!(out.read(), vec![0; out.len()]);
        assert_eq!(lhs.read(), vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(rhs.read(), vec![1, 2, 3, 4, 5, 6]);

        device
            .launch_kernel1d(
                lhs.len(),
                &add_src,
                "add",
                &[&lhs, &rhs, &mut out, &lhs.len()],
            )
            .unwrap();

        device
            .launch_kernel1d(
                lhs.len(),
                &add_src,
                "add",
                &[&out, &lhs, &mut rhs, &lhs.len()],
            )
            .unwrap();

        device
            .launch_kernel1d(
                rhs.len(),
                &mul_src,
                "mul",
                &[&out, &rhs, &mut lhs, &rhs.len()],
            )
            .unwrap();

        // device.run().unwrap();

        assert_eq!(out.read(), vec![2, 4, 6, 8, 10, 12]);
        assert_eq!(rhs.read(), vec![3, 6, 9, 12, 15, 18]);
        assert_eq!(lhs.read(), vec![6, 24, 54, 96, 150, 216]);
    }

    #[test]
    fn test_lazy_by_separate_module_cuda() {
        // let lazy = <Lazy<Base> as Module<CUDA>>::new();

        // let mut device = CUDA::<Base>::new(0).unwrap();
        // device.lazy_setup().unwrap();
    }

    fn cuda_ew<'a, Mods>(
        device: &'a CUDA<Mods>,
        lhs: &Buffer<i32, CUDA<Mods>>,
        rhs: &Buffer<i32, CUDA<Mods>>,
        src: String,
        fn_name: &'static str,
    ) -> Buffer<'a, i32, CUDA<Mods>>
    where
        Mods: 'static + AddOperation + Retrieve<CUDA<Mods>, i32, ()>,
    {
        let mut out = device.retrieve(lhs.len(), (lhs.id(), rhs.id())).unwrap();

        device
            .add_op((lhs, rhs, &mut out), move |(lhs, rhs, out)| {
                let device = lhs.device();
                device.launch_kernel1d(lhs.len(), &src, fn_name, &[lhs, rhs, out, &lhs.len()])
            })
            .unwrap();

        out
    }

    #[test]
    fn test_cuda_add_ew_op() {
        let device = CUDA::<Base>::new(0).unwrap();

        let lhs = device.buffer([1, 2, 3, 4, 5, 6]);
        let rhs = device.buffer([1, 2, 3, 4, 5, 6]);

        let out = cuda_ew(&device, &lhs, &rhs, ew_src("add", '+'), "add");
        assert_eq!(out.read(), [2, 4, 6, 8, 10, 12]);
    }

    #[test]
    fn test_cuda_lazy_retrieving_exec_op() {
        let device = CUDA::<Lazy<Base>>::new(0).unwrap();

        let lhs = device.buffer([1, 2, 3, 4, 5, 6]);
        let rhs = device.buffer([1, 2, 3, 4, 5, 6]);

        let out = cuda_ew(&device, &lhs, &rhs, ew_src("add", '+'), "add");
        let out2 = cuda_ew(&device, &out, &rhs, ew_src("add", '+'), "add");

        let _ = unsafe { device.run() };

        assert_eq!(out.replace().read(), [2, 4, 6, 8, 10, 12]);
        assert_eq!(out2.replace().read(), [3, 6, 9, 12, 15, 18]);
    }

    #[cfg(feature = "graph")]
    #[test]
    fn test_cuda_apply_fn_lazy() {
        let device = CUDA::<crate::Graph<Lazy<Base>>>::new(0).unwrap();

        let lhs = device.buffer([1., 2., 3., 4., 5., 6.]);
        let out = device.apply_fn(&lhs, |x| x.sin());
        let out = device.apply_fn(&out, |x| x.cos());
        let _out = device.apply_fn(&out, |x| x.ln());

        let _ = unsafe { device.run() };
    }
}
