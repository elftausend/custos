use std::any::Any;
use std::{marker::PhantomData, ffi::c_void, ptr::null_mut};


use std::{
    cell::RefCell,
    collections::HashMap,
    mem::{align_of, size_of},
    alloc::Layout
};

use custos::Node;
use custos::number::Number;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RawCpu {
    pub ptr: *mut u8,
    len: usize,
    align: usize,
    size: usize,
}

impl Drop for RawCpu {
    fn drop(&mut self) {
        unsafe {
            let layout = Layout::array::<u8>(self.len*self.size)
                .unwrap().align_to(self.align).unwrap();
            std::alloc::dealloc(self.ptr, layout);
        }
    }
}

#[derive(Debug)]
pub struct CPUCache {
    pub nodes: HashMap<Node, RawCpu>,
}

impl CPUCache {
    pub fn ret(&mut self) -> &CacheBuffer<f32> {
        
        todo!()
    }
    pub fn add_node<'a, T: Default + Copy>(&'a mut self, device: &'a CPU, node: Node) -> &'a CacheBuffer<'a, T> {
        let ptr: (*mut T, _, _) = device.alloc(node.len);
        
        self.nodes.insert(node, RawCpu {
            ptr: ptr.0 as *mut u8,
            len: node.len,
            align: align_of::<T>(),
            size: size_of::<T>(),
        });
        
        todo!()
        /*&CacheBuffer { 
            ptr,
            len: node.len,
            device: device.as_device(),
        }*/
    }

    pub fn get<'a, T: Default + Copy>(device: &'a CPU, len: usize) -> &CacheBuffer<T> {
        let node = Node::new(len);

        let mut cache = device.cache.borrow_mut();
        let buf_info_option = cache.nodes.get(&node);

        todo!()
        /*match buf_info_option {
            Some(buf_info) => {
                &CacheBuffer {
                    ptr: (buf_info.ptr as *mut T, null_mut(), 0),
                    len: buf_info.len,
                    device: device.as_device(),   
                }                    
            }
            None => cache.add_node(device, node),
        }*/
    
    }
}



#[derive(Debug)]
pub struct CPU {
    cache: RefCell<CPUCache>,
}

impl CPU {
    pub fn new() -> CPU {
        CPU {
            cache: RefCell::new(CPUCache { nodes: HashMap::new() })
        }
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self as *const CPU as *mut u8
    }
}

pub struct CudaDevice;

#[derive(Debug, Clone, Copy)]
pub enum DeviceType {
    CPU = 0,
    CUDA = 1,
    CL = 2,
}

pub trait Alloc<T> {
    fn alloc(&self, len: usize) -> (*mut T, *mut c_void, u64);
    fn as_device(&self) -> Device;
}

impl<T> Alloc<T> for CPU {
    fn alloc(&self, len: usize) -> (*mut T, *mut c_void, u64) {
        (null_mut(), null_mut(), 0)
    }

    fn as_device(&self) -> Device {
        todo!()
        //&Device {
        //    device_type: DeviceType::CPU,
        //    device: self as *const CPU as *mut u8
        //}
    }

}

impl<T> Alloc<T> for CudaDevice {
    fn alloc(&self, len: usize) -> (*mut T, *mut c_void, u64) {
        (null_mut(), null_mut(), 0)
    }

    fn as_device(&self) -> Device {
        todo!()
        //&Device {
        //    device_type: DeviceType::CUDA,
        //    device: self as *const CudaDevice as *mut u8
        //}
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Device {
    device_type: DeviceType,
    device: *mut u8
}

pub trait AddOp<T> {
    fn add(&self, lhs: &Buffer<T>, rhs: &Buffer<T>) -> Buffer<T>;
}


impl<T: Default + Copy> AddOp<T> for CPU {
    fn add(&self, lhs: &Buffer<T>, rhs: &Buffer<T>) -> Buffer<T> {
        let out = CPUCache::get::<T>(self, lhs.len);
        todo!()
        //out
    }
}

impl<T> AddOp<T> for CudaDevice {
    fn add(&self, lhs: &Buffer<T>, rhs: &Buffer<T>) -> Buffer<T> {
        println!("add");
        todo!()
    }
}

pub fn add_device_any<T: Default + Copy>(device: &dyn Any, device_type: DeviceType) -> &dyn AddOp<T> {
    unsafe {
        match device_type {
            DeviceType::CPU => device.downcast_ref::<CPU>().unwrap(),
            DeviceType::CUDA => device.downcast_ref::<CudaDevice>().unwrap(),
            DeviceType::CL => todo!(),
        }
    }
}

pub fn add_device<'a, T: Default + Copy>(device: Device) -> &'a dyn AddOp<T> {
    unsafe {
        match device.device_type {
            DeviceType::CPU => &*(device.device as *mut CPU) ,
            DeviceType::CUDA => &*(device.device as *mut CudaDevice),
            DeviceType::CL => todo!(),
        }
    }
}

pub fn alloc_device<'a, T: Default + Copy>(device: &Device) -> &'a dyn Alloc<T> {
    unsafe {
        match device.device_type {
            DeviceType::CPU => &*(device.device as *mut CPU) ,
            DeviceType::CUDA => &*(device.device as *mut CudaDevice),
            DeviceType::CL => todo!(),
        }
    }
}

pub struct CacheBuffer<'a, T> {
    device: &'a Device,
    ptr: (*mut T, *mut c_void, u64),
    len: usize,
}


pub struct Buffer<'a, T> {
    //device: &'a dyn Any,
    device: Device,
    device_type: DeviceType,
    ptr: (*mut T, *mut c_void, u64),
    len: usize,
    p: PhantomData<&'a T>,
}

impl<'a, T> Buffer<'a, T> {
    pub fn new<A: Alloc<T> + ?Sized>(device: &'a A, len: usize) -> Self {
        Buffer {
            //device: device,
            device: device.as_device(),
            device_type: device.as_device().device_type,
            ptr: device.alloc(len),
            len,
            p: PhantomData
        }
    }

    pub fn add(&self, rhs: &Buffer<'a, T>) -> Buffer<'a, T> where T: Default + Copy {
        // with dyn Any:
        //let device: &'b dyn AddOp<T> = add_device_any::<T>(self.device, self.device_type);

        let device: &dyn AddOp<T> = add_device::<T>(self.device);

        device.add(self, rhs)
    }
}

impl<'a, T> Drop for Buffer<'a, T> {
    fn drop(&mut self) {
        todo!()
    }
}

#[test]
fn test_structure() {
    let device = CPU::new();
    let lhs = Buffer::<f32>::new(&device, 100);
    let rhs = Buffer::<f32>::new(&device, 100);
    let out = lhs.add(&rhs);

    let out = {
        //let device = CPU::new();
        let lhs = Buffer::<f32>::new(&device, 100);
        let rhs = Buffer::<f32>::new(&device, 100);
        lhs.add(&rhs)
    };
}

#[test]
fn test() {
    let device = CPU::new();

    let buf = {
        Buffer::<f32>::new(&device, 10)
    };

    let buf = {
        let device = CPU::new();
        let a = Buffer::<f32>::new(&device, 10);
        let b = Buffer::<f32>::new(&device, 10);
        let c = a.add(&b);
        c
    };
    //buf.device;



    let buf = {
        
        let buf = {
            let device = Alloc::<f32>::as_device(&device);
            let alloc: &dyn Alloc<f32> = alloc_device(&device);
           // Buffer {
                
            //}
            Buffer::new(alloc, 10)
        };
        let device = Alloc::<f32>::as_device(&device);
        let alloc: &dyn Alloc<f32> = alloc_device(&device);
        let buf = Buffer::new(alloc, 10);
        buf.add(&buf)
        //let device = Alloc::<f32>::as_device(&device);
        //let alloc: &dyn Alloc<f32> = alloc_device(device);
        
    };
}

pub struct Owns<'a, T> {
    other: Buffer<'a, T>,
    buf: Option<&'a Buffer<'a, T>>
}


impl<'a, T: Number> Owns<'a, T> {
    pub fn test(&mut self, inputs: &'a Buffer<T>) {
        let mat: &Buffer<T> = self.buf.unwrap();
        self.other = mat.add(inputs).add(inputs);
    }  
}
