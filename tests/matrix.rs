use std::{rc::Rc, cell::RefCell};

use custos::{AsDev, libs::{cpu::CPU, opencl::{api::OCLError, CLDevice}}, Matrix, BaseOps, VecRead, Buffer};

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
pub struct RcCPU {
    ptrs: Vec<*mut usize>
}

impl RcCPU {
    pub fn new() -> Rc<RefCell<RcCPU>> {
        Rc::new(RefCell::new(RcCPU { ptrs: Vec::new() }))
    }

    pub fn buffer(&mut self, len: usize) -> Buffer<f32> {
        let buffer =  Buffer::<f32>::from( (&CPU, vec![1.12; len]) );
        let ptr = buffer.ptr as *mut usize;
        self.ptrs.push(ptr);
        buffer
    }
}

impl Drop for RcCPU {
    fn drop(&mut self) {
        for ptr in &self.ptrs {
            unsafe {    
                drop(Box::from_raw(*ptr));
            }
        }
        self.ptrs.clear()
    }
}

fn buffer<>(device: Rc<RefCell<RcCPU>>, len: usize) -> Buffer<f32> {
    device.borrow_mut().buffer(len)
}

#[test]
fn test_rccpu() {
    let device = RcCPU::new();
    
    let _ = buffer::<>(device.clone(), 10000000);
    let _ = buffer::<>(device.clone(), 1000000);

    drop(device);

    loop {
        
    }
}
