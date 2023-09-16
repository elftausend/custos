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

// impl Drop for LazyCudaGraph {
//     fn drop(&mut self) {
//         unsafe {
//             ManuallyDrop::drop(&mut self.graph_exec);
//             ManuallyDrop::drop(&mut self.graph);
//         }
//     }
// }

impl LazyCudaGraph {
    pub fn new(stream: &Stream) -> Result<Self, CudaErrorKind> {
        let graph = create_graph_from_captured_stream(stream)?;
        let graph_exec = ManuallyDrop::new(create_graph_execution(&graph)?);

        Ok(LazyCudaGraph { graph, graph_exec })
    }

    pub fn launch(&self, stream: super::api::CUstream) -> Result<(), CudaErrorKind> {
        unsafe { cuGraphLaunch(self.graph_exec.0.as_ptr(), stream).to_result()? }
        Ok(())
    }
}

#[cfg(feature = "lazy")]
impl<Mods> crate::LazyRun for CUDA<Mods> {
    #[inline]
    fn run(&self) -> crate::Result<()> {
        let graph = self
            .graph
            .get_or_init(|| LazyCudaGraph::new(&self.stream()).unwrap());
        
        graph.launch(self.stream.0)?;
        self.stream().sync()?;
        Ok(())
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
    use crate::{Base, Device, Lazy, LazyRun, CUDA};

    pub fn ew_src(fn_name: &str, operator: char) -> String {
        format!(r#"
            extern "C" __global__ void {fn_name}(int* lhs, int* rhs, int* out, int len) {{
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= len) {{
                    return;
                }}
                out[idx] = lhs[idx] {operator} rhs[idx];
            }}
        "#)
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
            .launch_kernel1d(lhs.len(), &add_src, "add", &[&lhs, &rhs, &mut out, &lhs.len()])
            .unwrap();
        
        device
            .launch_kernel1d(lhs.len(), &add_src, "add", &[&out, &lhs, &mut rhs, &lhs.len()])
            .unwrap();

        device
            .launch_kernel1d(rhs.len(), &mul_src, "mul", &[&out, &rhs, &mut lhs, &rhs.len()])
            .unwrap();

        assert_eq!(out.read(), vec![0; out.len()]);
        assert_eq!(lhs.read(), vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(rhs.read(), vec![1, 2, 3, 4, 5, 6]);

        device.run().unwrap();

        assert_eq!(out.read(), vec![2, 4, 6, 8, 10, 12]);
        assert_eq!(rhs.read(), vec![3, 6, 9, 12, 15, 18]);
        assert_eq!(lhs.read(), vec![6, 24, 54, 96, 150, 216]);
    }
}
