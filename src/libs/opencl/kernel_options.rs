use crate::{matrix::Matrix, number::Number};

use super::{api::{set_kernel_arg, enqueue_nd_range_kernel, OCLError}, cl_cache::{Node}, CLDevice, GenericOCL, CLCache, CL_CACHE};

pub struct KernelArg<T> {
    tensor: Option<Matrix<T>>,
    number: Option<T>,
}
impl <T>KernelArg<T> {
    pub fn new(tensor: Option<Matrix<T>>, number: Option<T>) -> KernelArg<T> {
        KernelArg {
            tensor,
            number
        }
    }
}

pub trait TKernelArg<T> {
    fn is_number(&self) -> bool;
    fn as_karg(&self) -> KernelArg<T>;
}


impl <T: Copy>TKernelArg<T> for Matrix<T> {
    fn is_number(&self) -> bool {
        false
    }

    fn as_karg(&self) -> KernelArg<T> {
        KernelArg::new(Some(*self), None)
    }
}

impl <T: Copy>TKernelArg<T> for &Matrix<T> {
    fn is_number(&self) -> bool {
        false
    }

    fn as_karg(&self) -> KernelArg<T> {
        KernelArg::new(Some(**self), None)
    }
}

impl <T: Number>TKernelArg<T> for T {
    fn is_number(&self) -> bool {
        true
    }

    fn as_karg(&self) -> KernelArg<T> {
        KernelArg::new(None, Some(*self))
    }
}

pub struct KernelOptions<'a, T> {
    src: &'a str,
    lhs: Matrix<T>,
    rhs: Option<Matrix<T>>,
    output: Option<Matrix<T>>,
    tensor_args: Vec<(Matrix<T>, usize)>,
    number_args: Vec<(T, usize)>,
    gws: [usize; 3],
    lws: Option<[usize; 3]>,
    wd: usize,
    device: CLDevice,
}

impl <'a, T: GenericOCL>KernelOptions<'a, T> {
    pub fn new(device: CLDevice, lhs: Matrix<T>, gws: [usize; 3], src: &'a str) -> KernelOptions<'a, T> {
        let wd;
        if gws[0] == 0 {
            panic!("wrong gws")
        } else if gws[1] == 0 {
            wd=1;
        } else if gws[2] == 0 {
            wd=2;
        } else {
            wd=3;
        }
        let tensor_args = vec![(lhs, 0)];
        KernelOptions {
            src,
            lhs,
            rhs: None,
            output: None,
            tensor_args,
            number_args: Vec::new(),
            gws,
            lws: None,
            wd,
            device,
        }
    }
    pub fn with_lws(&mut self, lws: [usize; 3]) -> &mut KernelOptions<'a, T> {
        self.lws = Some(lws);
        self
    }
    pub fn with_rhs(&mut self, rhs: Matrix<T>) -> &mut KernelOptions<'a, T> {
        self.tensor_args.push((rhs, 1));
        self.rhs = Some(rhs);
        self
    }
    /*
    pub fn with_pre_output(&mut self, output: &'a Tensor<T>) -> &mut KernelOptions<'a, T> {
        let output = output.copy_or_move();
        /* 
        let mem = Box::new(*output.get_mem().downcast_ref::<Mem>().unwrap());
        let output = Tensor {
            ts: output.ts,
            mem,
            backend: output.backend.clone(),
            _pd: PhantomData,
        };
        */
        self.output = Some(output);
        self

    }
    */
    pub fn add_arg<A: TKernelArg<T>>(&'a mut self, arg: &'a A) -> &mut KernelOptions<'a, T> {
        let idx = self.number_args.len()+self.tensor_args.len();
        let karg = arg.as_karg();
        //could be checked with is_some on karg.number, or match
        if arg.is_number() {
            self.number_args.push((karg.number.unwrap(), idx));
        } else {
            self.tensor_args.push((karg.tensor.unwrap(), idx));
        }

        self
        
    }
    pub fn with_output(&mut self, out_dims: (usize, usize)) -> &mut KernelOptions<'a, T> {
        self.output = Some(CLCache::get(Node::new(out_dims)));

        /* 
        match self.rhs {
            Some(rhs) => {
                let output = OCLCache::get(Node::new(out_dims));
                self.output = Some(output)
            },
            None => {
                self.output = Some(OCLCache::get_output_cache(self.backend.clone(), self.lhs, self.lhs, with_op_or_not));
            },
        }
        */
        self
    }
    pub fn run(&'a mut self) -> Result<Matrix<T>, OCLError> {
        let device = self.device;
        
        let kernel = unsafe {CL_CACHE.arg_kernel_cache(device, &self.tensor_args, &self.number_args, self.output, self.src.to_string())};
        
        
        for index in 0..self.number_args.len() {
            let arg = self.number_args.get(index).unwrap();
            set_kernel_arg(&kernel, arg.1, &arg.0)
        }

        enqueue_nd_range_kernel(&device.get_queue(), &kernel, self.wd, &self.gws, self.lws.as_ref(), None)?;
        
        match self.output {
            Some(out) => {
                Ok(out)
            },
            None => {
                Ok(self.lhs)
                
            },
        }
    
    }
}
