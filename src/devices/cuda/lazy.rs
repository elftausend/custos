use core::ptr::NonNull;

use crate::CUDA;

use super::api::{cuStreamEndCapture, Graph, CudaErrorKind, GraphExec, cuGraphInstantiate, cuGraphLaunch};


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

pub(super) struct LazyCudaGraph {
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
}

#[cfg(feature = "lazy")]
impl<Mods> crate::LazyRun for CUDA<Mods> {
    #[inline]
    fn run(&mut self) -> crate::Result<()> {
        if self.graph.is_none() {
            self.graph = Some(LazyCudaGraph::new(self.stream.0)?);
        }
        let graph = self.graph.as_ref().unwrap(); 
        unsafe { cuGraphLaunch(graph.graph_exec.0.as_ptr(), self.stream.0).to_result()? } 
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{CUDA, Lazy, Base, Device};

    #[test]
    fn test_lazy_cuda_run() {
        let device = CUDA::<Lazy<Base>>::new(0).unwrap();
        let lhs = device.buffer([1, 2, 3, 4, 5, 6]);
        let rhs = device.buffer([1, 2, 3, 4, 5, 6]);

        let src = r#"
            extern "C" __global__ void add(int* lhs, int* rhs, int len) {
                
            }
        "#;
    }
}
