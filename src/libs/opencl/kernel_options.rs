
/* 
pub struct KernelArg<'a, T> {
    tensor: Option<&'a Tensor<T>>,
    number: Option<T>,
}
impl <'a, T>KernelArg<'a, T> {
    pub fn new(tensor: Option<&'a Tensor<T>>, number: Option<T>) -> KernelArg<'a, T> {
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

impl <T>TKernelArg<T> for &Tensor<T> {
    fn is_number(&self) -> bool {
        false
    }

    fn as_karg(&self) -> KernelArg<T> {
        KernelArg::new(Some(self), None)
    }
}

impl <T>TKernelArg<T> for Tensor<T> {
    fn is_number(&self) -> bool {
        false
    }

    fn as_karg(&self) -> KernelArg<T> {
        KernelArg::new(Some(self), None)
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
    lhs: &'a Tensor<T>,
    rhs: Option<&'a Tensor<T>>,
    output: Option<Tensor<T>>,
    tensor_args: Vec<(&'a Tensor<T>, usize)>,
    number_args: Vec<(T, usize)>,
    gws: [usize; 3],
    lws: Option<[usize; 3]>,
    wd: usize,
    backend: Backend<OpenCL>
}

impl <'a,T: Number>KernelOptions<'a, T> {
    pub fn new(backend: Backend<OpenCL>, lhs: &'a Tensor<T>, gws: [usize; 3], src: &'a str) -> KernelOptions<'a, T> {
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
            backend,
        }
    }
    pub fn with_lws(&mut self, lws: [usize; 3]) -> &mut KernelOptions<'a, T> {
        self.lws = Some(lws);
        self
    }
    pub fn with_rhs(&mut self, rhs: &'a Tensor<T>) -> &mut KernelOptions<'a, T> {
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
    pub fn with_output<WOP: TWithOpOrNot>(&mut self, with_op_or_not: WOP) -> &mut KernelOptions<'a, T> {
        match self.rhs {
            Some(rhs) => {
                let output = OCLCache::get_output_cache(self.backend.clone(),self.lhs, rhs, with_op_or_not);
                self.output = Some(output)
            },
            None => {
                self.output = Some(OCLCache::get_output_cache(self.backend.clone(), self.lhs, self.lhs, with_op_or_not));
            },
        }

        self
    }
    pub fn run(&'a mut self) -> Result<Tensor<T>, OCLError> {
        let device = self.backend.device();
        #[cfg(not(feature = "nocache"))]
        let kernel = unsafe {OCLCACHE.arg_kernel_cache(self.backend.clone(), &self.tensor_args, &self.number_args, self.output.as_ref(), self.src.to_string())};
        
        #[cfg(feature = "nocache")]
        let kernel = unsafe {OCLCACHE.nc_kernel_cache(self.src.to_string(), self.backend.framework.device_idx)};

        #[cfg(feature = "nocache")]
        for index in 0..self.tensor_args.len() {
            let arg = self.tensor_args.get(index).unwrap();
            set_kernel_arg(&kernel, arg.1, &arg.0.get_mem().downcast_ref::<Mem>().unwrap().0)

        }

        for index in 0..self.number_args.len() {
            let arg = self.number_args.get(index).unwrap();
            set_kernel_arg(&kernel, arg.1, &arg.0)
        }

        #[cfg(feature = "nocache")] {
            use crate::backend::TBackend;
            let idx = self.number_args.len()+self.tensor_args.len();
            match &mut self.output {
                Some(out) => {
   
                    set_kernel_arg(&kernel, idx, &out.get_mem().downcast_ref::<Mem>().unwrap().0);
                    enqueue_nd_range_kernel(&device.get_queue(), &kernel, self.wd, &self.gws, self.lws.as_ref(), None)?;
                    
                    let mem = out.get_mut_mem().downcast_mut::<Mem>().unwrap();
                    let mem = mem.as_cloned_no_drop();
                    Ok(Tensor::from_mem( out.ts, mem))
                    
                }
                None => {
                    enqueue_nd_range_kernel(&device.get_queue(), &kernel, self.wd, &self.gws, self.lws.as_ref(), None)?;
                    Ok(self.backend.as_cloned(self.lhs))
                }
            }
        }
        
        #[cfg(not(feature = "nocache"))] {
            enqueue_nd_range_kernel(&device.get_queue(), &kernel, self.wd, &self.gws, self.lws.as_ref(), None)?;
            
            match &self.output {
                Some(out) => {
                    let mem = *out.get_mem().downcast_ref::<Mem>().unwrap();
                    Ok(Tensor::from_mem(out.ts, mem))
                },
                None => {
                    let mem = *self.lhs.get_mem().downcast_ref::<Mem>().unwrap();
                    Ok(Tensor::from_mem(self.lhs.ts, mem))
                },
            }
        }
    }
}
*/