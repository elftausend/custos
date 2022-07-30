use std::{marker::PhantomData, ffi::c_void, ptr::null_mut};


use std::{
    cell::RefCell,
    collections::HashMap,
    mem::{align_of, size_of},
    alloc::Layout
};

use custos::Node;

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

thread_local! {
    pub static CPU_CACHE: RefCell<CPUCache> = RefCell::new(CPUCache {nodes: HashMap::new()});
}

pub struct CPUCache {
    pub nodes: HashMap<Node, RawCpu>,
}

impl CPUCache {
    pub fn add_node<'a, T: Default + Copy>(&mut self, device: &'a CPU, node: Node) -> Buffer<'a, T> {
        let ptr: (*mut T, _, _) = device.alloc(node.len);
        
        self.nodes.insert(node, RawCpu {
            ptr: ptr.0 as *mut u8,
            len: node.len,
            align: align_of::<T>(),
            size: size_of::<T>(),
        });
        
        Buffer {
            ptr,
            len: node.len,
            device: device.as_device(),
            p: PhantomData,
        }
    }

    pub fn get<'a, T: Default + Copy>(device: &'a CPU, len: usize) -> Buffer<'a, T> {
        let node = Node::new(len);
        CPU_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let buf_info_option = cache.nodes.get(&node);

            match buf_info_option {
                Some(buf_info) => {
                    Buffer {
                        ptr: (buf_info.ptr as *mut T, null_mut(), 0),
                        len: buf_info.len,
                        device: device.as_device(),
                        p: PhantomData,
                    }                    
                }
                None => cache.add_node(device, node),
            }
        })
    }
}



#[derive(Debug, Clone)]
pub struct CPU;

pub struct CudaDevice;

pub enum DeviceType {
    CPU = 0,
    CUDA = 1,
    CL = 2,
}

pub trait Alloc {
    fn alloc<T>(&self, len: usize) -> (*mut T, *mut c_void, u64);
    fn as_device(&self) -> Device;
}

impl Alloc for CPU {
    fn alloc<T>(&self, len: usize) -> (*mut T, *mut c_void, u64) {
        (null_mut(), null_mut(), 0)
    }

    fn as_device(&self) -> Device {
        Device {
            device_type: DeviceType::CPU,
            device: self as *const CPU as *mut u8
        }
    }

}

impl Alloc for CudaDevice {
    fn alloc<T>(&self, len: usize) -> (*mut T, *mut c_void, u64) {
        (null_mut(), null_mut(), 0)
    }

    fn as_device(&self) -> Device {
        Device {
            device_type: DeviceType::CUDA,
            device: self as *const CudaDevice as *mut u8
        }
    }
}

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
        out
    }
}

impl<T> AddOp<T> for CudaDevice {
    fn add(&self, lhs: &Buffer<T>, rhs: &Buffer<T>) -> Buffer<T> {
        println!("add");
        todo!()
    }
}

pub fn add_device<T: Default + Copy>(device: &Device) -> &dyn AddOp<T> {
    unsafe {
        match device.device_type {
            DeviceType::CPU => &*(device.device as *mut CPU) ,
            DeviceType::CUDA => &*(device.device as *mut CudaDevice),
            DeviceType::CL => todo!(),
        }
    }
}


pub struct Buffer<'a, T> {
    device: Device,
    ptr: (*mut T, *mut c_void, u64),
    len: usize,
    p: PhantomData<&'a T>
}

impl<'a, T> Buffer<'a, T> {
    pub fn new<A: Alloc>(device: &'a A, len: usize) -> Self {
        Buffer {
            device: device.as_device(),
            ptr: device.alloc(len),
            len,
            p: PhantomData
        }
    }

    pub fn add(&self, rhs: &Buffer<T>) -> Buffer<T> where T: Default + Copy {
        let device = add_device::<T>(&self.device);
        device.add(self, rhs)
    }
}

#[test]
fn test_structure() {
    let device = CPU;
    let lhs = Buffer::<f32>::new(&device, 100);
    let rhs = Buffer::<f32>::new(&device, 100);
    let out = lhs.add(&rhs);

    let out = {
        let device = CPU;
        let lhs = Buffer::<f32>::new(&device, 100);
        let rhs = Buffer::<f32>::new(&device, 100);
        lhs.add(&rhs)
    };



}
