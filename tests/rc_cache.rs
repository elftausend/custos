use custos::{cpu::RawCpu, number::Number, AsDev, BufFlag, Buffer, Device, Node, CPU};
use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    ffi::c_void,
    mem::align_of,
    ptr::null_mut,
    rc::{Rc, Weak},
};

thread_local! {
    pub static CACHE: RefCell<Cache> = RefCell::new(Cache {nodes: HashMap::new()});
}

pub struct Cache {
    pub nodes: HashMap<Node, Rc<RawCpu>>,
}

pub struct CacheBuffer<T> {
    buf: Cell<Buffer<T>>,
    ptr: (Weak<RawCpu>, *mut c_void, u64),
}
impl<T> CacheBuffer<T> {
    pub fn new(ptr: (Weak<RawCpu>, *mut c_void, u64), len: usize) -> CacheBuffer<T> {
        CacheBuffer {
            buf: Cell::new(Buffer {
                ptr: (null_mut(), null_mut(), 0),
                len,
                flag: BufFlag::Cache,
            }),
            ptr,
        }
    }

    /*fn to_buf(self) -> Buffer<T> {
        let ptr = ( (*Rc::as_ref(&self.ptr.0.upgrade().expect("invalid ptr !!!"))).ptr as *mut T, self.ptr.1, self.ptr.2);
        Buffer { ptr, len: ..., flag: BufFlag::Cache }
    }*/

    fn as_buf(&self) -> &Buffer<T> {
        let ptr = (
            (*Rc::as_ref(&self.ptr.0.upgrade().expect("invalid ptr !!!"))).ptr as *mut T,
            self.ptr.1,
            self.ptr.2,
        );
        let buf = self.buf.as_ptr();
        unsafe {
            (*buf).ptr = ptr;
            &*buf
        }
    }
}

impl<T> std::ops::Deref for CacheBuffer<T> {
    type Target = Buffer<T>;

    fn deref(&self) -> &Self::Target {
        self.as_buf()
    }
}

impl<T> std::ops::DerefMut for CacheBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        let ptr = (
            (*Rc::as_ref(&self.ptr.0.upgrade().expect("invalid ptr !!!"))).ptr as *mut T,
            self.ptr.1,
            self.ptr.2,
        );
        let buf = self.buf.as_ptr();
        unsafe {
            (*buf).ptr = ptr;
            &mut *buf
        }
    }
}

impl Cache {
    pub fn add_node<T: Default + Copy>(&mut self, device: &CPU, node: Node) -> CacheBuffer<T> {
        let ptr: (*mut T, _, _) = device.alloc(node.len);

        let raw_cpu = Rc::new(RawCpu {
            ptr: ptr.0 as *mut usize,
            len: node.len,
            align: align_of::<T>(),
        });
        let cb = CacheBuffer::new((Rc::downgrade(&raw_cpu), null_mut(), 0), node.len);

        self.nodes.insert(node, raw_cpu);

        cb
    }

    pub fn get<T: Default + Copy>(device: &CPU, len: usize) -> CacheBuffer<T> {
        //assert!(!device.cpu.borrow().ptrs.is_empty(), "no cpu allocations");

        let node = Node::new(len);
        CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let buf_info_option = cache.nodes.get(&node);

            match buf_info_option {
                Some(buf_info) => {
                    CacheBuffer::new((Rc::downgrade(buf_info), null_mut(), 0), node.len)
                }
                None => cache.add_node(device, node),
            }
        })
    }
}

fn deref_buf<T: Number>(buf: &mut Buffer<T>) {
    for value in buf {
        *value += T::one();
    }
}

#[test]
fn test_rc_cache() {
    let device = CPU::new().select();

    let mut buf = Cache::get::<f32>(&device, 100);
    deref_buf(&mut buf);

    println!("buf: {:?}", buf.as_buf());

    CACHE.with(|cache| cache.borrow_mut().nodes.clear());
    println!("buf: {:?}", buf.as_buf());
}
