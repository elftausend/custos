use std::{rc::Rc, cell::RefCell};

use custos::{AsDev, libs::{cpu::CPU, opencl::{api::OCLError, CLDevice}}, Matrix, BaseOps, VecRead, Buffer, Device, number::Number, range, Dealloc};

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
pub struct A {
    pub cpu: Rc<RefCell<RcCPU>>
}
impl A {
    pub fn new(cpu: Rc<RefCell<RcCPU>>) -> A {
        A { cpu }
    }
}

#[derive(Debug, Clone)]
pub struct RcCPU {
    pub ptrs: Vec<*mut usize>
}

impl RcCPU {
    pub fn new() -> A {
        A::new(Rc::new(RefCell::new(RcCPU { ptrs: Vec::new() })))
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

impl <T: Copy+Default>Device<T> for A {
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

impl <T: Copy+Default>VecRead<T> for A {
    fn read(&self, buf: Buffer<T>) -> Vec<T> {
        CPU.read(buf)
    }
}

impl <T: Number>BaseOps<T> for A {
    fn add(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        CPU.add(lhs, rhs)
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
        CPU.dealloc_cache();
        for ptr in &self.ptrs {
            unsafe {    
                drop(Box::from_raw(*ptr));
            }
        }
        self.ptrs.clear();
        println!("self: {}", self.ptrs.len())
    }
}



#[test]
fn test_rccpu() {
    let device = RcCPU::new();
    
    //let a = Matrix::<f32>::from( (device.clone(), (2, 3), &[1., 2., 3., 4., 5., 6.,]) );
    //let b = Matrix::<f32>::from( (device.clone(), (2, 3), &[1., 2., 3., 4., 5., 6.,]) );
    let a = Matrix::<i128>::new(device.clone(), (10000, 1000));
    let b = Matrix::new(device.clone(), (10000, 1000));

    for _ in range(100) {
        device.add(a, b);
    }
    println!("fin");
    drop(device);


    /*

    let a = buffer::<>(device.clone(), 10000000);
    let _ = buffer::<>(device.clone(), 1000000);

    //drop(device);

    for _ in 0..1000 {
        device.borrow().read(a);
    }

    */
}