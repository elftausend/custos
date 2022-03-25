use std::{rc::Rc, cell::RefCell};

use custos::{AsDev, libs::{cpu::{CPU, ew_op, CPU2}, opencl::{api::OCLError, CLDevice}}, Matrix, BaseOps, VecRead, Buffer, Device, number::Number, range, Dealloc};

#[test]
fn test_dealloc_cpu() {
    let device = CPU2::new();
    
    let a = Matrix::from( (device.clone(), (2, 2), &[0.25, 0.5, 0.75, 1.] ) );
    let b = Matrix::from( (device.clone(), (2, 2), &[1., 2., 3., 4.,] ) );

    device.add(a, b);
}

#[test]
fn test_matrix_read() {
    CPU.select();

    let matrix = Matrix::from(((2, 3), &[1.51, 6.123, 7., 5.21, 8.62, 4.765]));
    let read = matrix.read();
    assert_eq!(&read, &[1.51, 6.123, 7., 5.21, 8.62, 4.765]);
}

//safety ideas
#[test]
fn test_simple() -> Result<(), OCLError> {
    //device: Rc<>
    //when dropped: deallocate ?
    CPU.select();

    let a = Matrix::from( ((2, 3), &[1, 2, 3, 4, 5, 6] ));

    // "drop(device)" : a is still allocated

    let b = Matrix::from( ((2, 3), &[6, 5, 4, 3, 2, 1] ));

    let c = a + b;
    assert_eq!(c.read(), vec![7, 7, 7, 7, 7, 7]);

    let a = Matrix::from( (CPU, (2, 2), &[0.25, 0.5, 0.75, 1.] ) );
    let b = Matrix::from( (CPU, (2, 2), &[1., 2., 3., 4.,] ) );

    let c_cpu = CPU.mul(a, b);
    assert_eq!(CPU.read(c_cpu.data()), vec![0.25, 1., 2.25,  4.,]);

    CLDevice::get(0)?.select();

    let a = Matrix::from( ((2, 2), &[0.25f32, 0.5, 0.75, 1.] ) );
    let b = Matrix::from( ((2, 2), &[1., 2., 3., 4.,] ) );

    let c_cl = a * b;
    assert_eq!(CPU.read(c_cpu.data()), c_cl.read());

    Ok(())
}

#[derive(Debug, Clone)]
pub struct Cldev {
    pub cl: Rc<RefCell<RcCL>>
}
impl Cldev {
    pub fn new(cl: Rc<RefCell<RcCL>>) -> Cldev {
         Cldev { cl }
    }
}

#[derive(Debug, Clone)]
pub struct RcCL {
    pub ptrs: Vec<*mut usize>
}

#[derive(Debug, Clone)]
pub struct Cpu {
    pub cpu: Rc<RefCell<RcCPU>>
}
impl Cpu {
    pub fn new(cpu: Rc<RefCell<RcCPU>>) -> Cpu {
        Cpu { cpu }
    }
}

#[derive(Debug, Clone)]
pub struct RcCPU {
    pub ptrs: Vec<*mut usize>
}

impl RcCPU {
    pub fn new() -> Cpu {
        Cpu::new(Rc::new(RefCell::new(RcCPU { ptrs: Vec::new() })))
    }

    pub fn buffer(&mut self, len: usize) -> Buffer<f32> {
        let buffer = Buffer::<f32>::from( (&CPU, vec![1.12; len]) );
        let ptr = buffer.ptr as *mut usize;
        self.ptrs.push(ptr);
        buffer
    }

    pub fn read(&self, buf: Buffer<f32>) -> Vec<f32> {
        CPU.read(buf)
    }
}

impl <T: Copy+Default>Device<T> for Cpu {
    fn alloc(&self, len: usize) -> *mut T {
        let ptr = CPU.alloc(len);
        self.cpu.borrow_mut().ptrs.push(ptr as *mut usize);
        ptr
    }

    fn with_data(&self, data: &[T]) -> *mut T {
        let ptr = CPU.with_data(data);
        self.cpu.borrow_mut().ptrs.push(ptr as *mut usize);
        ptr
    }
}

impl <T: Copy+Default>VecRead<T> for Cpu {
    fn read(&self, buf: Buffer<T>) -> Vec<T> {
        CPU.read(buf)
    }
}

impl <T: Number>BaseOps<T> for Cpu {
    fn add(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        ew_op(lhs, rhs, | x, y| x+y)
    }

    fn sub(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        CPU.sub(lhs, rhs)
    }

    fn mul(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        CPU.mul(lhs, rhs)
    }
}


impl Drop for RcCPU {
    fn drop(&mut self) {
        CPU::dealloc_cache();
        for ptr in &self.ptrs {
            unsafe {    
                drop(Box::from_raw(*ptr));
            }
        }
        self.ptrs.clear();
        println!("self: {}", self.ptrs.len())
    }
}


#[derive(Debug, Clone)]
pub struct Dev2 {
    pub cl_device: Option<Cldev>,
    pub cpu: Option<Cpu>,
}   

impl Dev2 {
    pub fn new(cl_device: Option<Cldev>, cpu: Option<Cpu>) -> Dev2 {
        Dev2 { cl_device, cpu }
    }
}

thread_local! {
    pub static GDEVICE: RefCell<Dev2> = RefCell::new(Dev2 { cl_device: None, cpu: None });
}

pub trait AsDev2 {
    fn as_dev(&self) -> Dev2;
    ///selects self as global device
    fn select(self) -> Self where Self: AsDev2+Clone {
        let dev = self.as_dev();
        GDEVICE.with(|d| *d.borrow_mut() = dev);        
        self
    }
}

pub fn get_device<T: Default+Copy>() -> Box<dyn Device<T>> {
    GDEVICE.with(|d| {
        let dev = d.borrow();
        match &dev.cl_device {
            Some(cl) => todo!()/*Box::new(cl.clone())*/,
            None => Box::new(dev.cpu.clone().unwrap()),
    }})
}

#[macro_export]
macro_rules! get_device2 {
    
    ($t:ident, $g:ident) => {    
        {     
            let dev: Box<dyn $t<$g>> = GDEVICE.with(|d| {
                let dev = d.borrow();
                match &dev.cl_device {
                    Some(_) => todo!()/*Box::new(cl.clone())*/,
                    None => Box::new(dev.cpu.clone().unwrap()),
                }
            });
            dev
        }
    }
}

impl AsDev2 for Cpu {
    fn as_dev(&self) -> Dev2 {
        Dev2::new(None, Some(self.clone()))
    }
}

impl AsDev2 for Cldev {
    fn as_dev(&self) -> Dev2 {
        Dev2::new(Some(self.clone()), None)
    }
}


fn test_rccpu() {
    let device = RcCPU::new();
    
    //let a = Matrix::<f32>::from( (device.clone(), (2, 3), &[1., 2., 3., 4., 5., 6.,]) );
    //let b = Matrix::<f32>::from( (device.clone(), (2, 3), &[1., 2., 3., 4., 5., 6.,]) );
    let a = Matrix::<i128>::new(device.clone(), (10000, 1000));
    let b = Matrix::new(device.clone(), (10000, 1000));

    for _ in range(50) {
        device.add(a, b);
    }

    drop(device);
    
}

#[test]
fn test_rccpu_2() {
    let device = RcCPU::new().select();

    let a = Matrix::<f32>::from( (device.clone(), (2, 3), &[1., 2., 3., 4., 5., 6.,]) );
    let b = Matrix::<f32>::from( (device.clone(), (2, 3), &[1., 2., 3., 4., 5., 6.,]) );

    let dev = get_device2!(BaseOps, f32);
    let result = dev.add(a, b);

    assert_eq!(device.read(result.data()), &[2., 4., 6., 8., 10., 12.]);   
}