use core::ptr::NonNull;

use crate::CUDA;

use super::api::{cuStreamEndCapture, Graph, CudaErrorKind, GraphExec, cuGraphInstantiate, cuGraphLaunch, cuStreamBeginCapture, CUStreamCaptureMode};


fn create_graph_from_captured_stream(stream: super::api::CUstream) -> Result<Graph, CudaErrorKind> {
    let mut graph = std::ptr::null_mut();
    unsafe { cuStreamEndCapture(stream, &mut graph) }.to_result()?;
    
    Ok(Graph(NonNull::new(graph).ok_or(CudaErrorKind::ErrorStreamCaptureInvalidated)?))
}

fn create_graph_execution(graph: &Graph) -> Result<GraphExec, CudaErrorKind> {
    let mut graph_exec = std::ptr::null_mut();
    unsafe { cuGraphInstantiate(&mut graph_exec, graph.0.as_ptr())}.to_result()?;
    Ok(GraphExec(NonNull::new(graph_exec).ok_or(CudaErrorKind::NotInitialized)?))
}

pub struct LazyCudaGraph {
    graph: Graph,
    graph_exec: GraphExec
}

impl LazyCudaGraph {
    pub fn new(stream: super::api::CUstream) -> Result<Self, CudaErrorKind> {
        let graph = create_graph_from_captured_stream(stream)?;
        let graph_exec = create_graph_execution(&graph)?;

        Ok(LazyCudaGraph {
            graph,
            graph_exec
        })
    }

    pub fn launch(&self, stream: super::api::CUstream) -> Result<(), CudaErrorKind> {
        unsafe { cuGraphLaunch(self.graph_exec.0.as_ptr(), stream).to_result()? } 
        Ok(())
    }
}

#[cfg(feature = "lazy")]
impl<Mods> crate::LazyRun for CUDA<Mods> {
    #[inline]
    fn run(&mut self) -> crate::Result<()> {
        if self.graph.is_none() {
            self.graph = Some(LazyCudaGraph::new(self.stream.0)?);
        }
        let graph = self.graph.as_ref().unwrap(); 
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
        }.to_result()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{CUDA, Lazy, Base, Device, Cached, CPU};

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

        device.launch_kernel1d(lhs.len(), src, "add", &[&lhs, &rhs, &mut out, &lhs.len()]).unwrap();

        assert_eq!(out.read(), vec![0; out.len()])
    }
}
