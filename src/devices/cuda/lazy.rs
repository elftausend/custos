use core::{mem::ManuallyDrop, ptr::NonNull};

use crate::CUDA;

use super::api::{
    cuGraphInstantiate, cuGraphLaunch, cuStreamBeginCapture, cuStreamEndCapture,
    CUStreamCaptureMode, CUgraphInstantiate_flags, CudaErrorKind, Graph, GraphExec, Stream,
};

fn create_graph_from_captured_stream(stream: &Stream) -> Result<Graph, CudaErrorKind> {
    let mut graph = std::ptr::null_mut();
    unsafe { cuStreamEndCapture(stream.0, &mut graph) }.to_result()?;

    Ok(Graph(
        NonNull::new(graph).ok_or(CudaErrorKind::ErrorStreamCaptureInvalidated)?,
    ))
}

fn create_graph_execution(graph: &Graph) -> Result<GraphExec, CudaErrorKind> {
    let mut graph_exec = std::ptr::null_mut();
    unsafe {
        cuGraphInstantiate(
            &mut graph_exec,
            graph.0.as_ptr(),
            CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        )
    }
    .to_result()?;
    Ok(GraphExec(
        NonNull::new(graph_exec).ok_or(CudaErrorKind::NotInitialized)?,
    ))
}

pub struct LazyCudaGraph {
    graph_exec: ManuallyDrop<GraphExec>,
    graph: ManuallyDrop<Graph>,
}

impl Drop for LazyCudaGraph {
    fn drop(&mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.graph_exec);
            ManuallyDrop::drop(&mut self.graph);
        }
    }
}

impl LazyCudaGraph {
    pub fn new(stream: &Stream) -> Result<Self, CudaErrorKind> {
        let graph = ManuallyDrop::new(create_graph_from_captured_stream(stream).unwrap());
        let graph_exec = ManuallyDrop::new(create_graph_execution(&graph).unwrap());

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
    use crate::{Base, Cached, Device, Lazy, LazyRun, CPU, CUDA};

    #[test]
    // #[ignore]
    fn test_lazy_cuda_run() {
        let device = CUDA::<Lazy<Base>>::new(0).unwrap();
        // let lhs = crate::Buffer::<i32, _>::new(&device, 100);
        // let rhs = crate::Buffer::<i32, _>::new(&device, 100);
        let lhs = device.buffer([1, 2, 3, 4, 5, 6]);
        let rhs = device.buffer([1, 2, 3, 4, 5, 6]);
        let mut out = lhs.empty_like();

        let src = r#"
            extern "C" __global__ void add(int* lhs, int* rhs, int* out, int len) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= len) {
                    return;
                }
                out[idx] = lhs[idx] + rhs[idx];
            }
        "#;

        device
            .launch_kernel1d(lhs.len(), src, "add", &[&lhs, &rhs, &mut out, &lhs.len()])
            .unwrap();

        assert_eq!(out.read(), vec![0; out.len()]);

        device.run().unwrap();

        assert_eq!(out.read(), vec![2, 4, 6, 8, 10, 12]);
        println!("fin")
    }
}
