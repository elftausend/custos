#[cfg(feature="opencl")]
use std::ffi::c_void;



use custos::{Buffer, VecRead};
#[cfg(feature="opencl")]
#[cfg(not(feature="safe"))]
use custos::{CPU, opencl::cl_tew};
#[cfg(feature="opencl")]
use custos::{opencl::api::{clCreateBuffer, MemFlags, OCLErrorKind}, InternCLDevice};
#[cfg(feature="opencl")]
use custos::{CLDevice, Error};

#[cfg(not(feature="safe"))]
#[cfg(feature="opencl")]
use std::ptr::null_mut;


#[cfg(feature="opencl")]
pub fn unified_mem<T>(device: &InternCLDevice, arr: &mut [T]) -> Result<*mut c_void, Error>{
    let mut err = 0;

    let r = unsafe {clCreateBuffer(device.ctx().0, MemFlags::MemReadWrite | MemFlags::MemCopyHostPtr, arr.len()*core::mem::size_of::<T>(), arr.as_mut_ptr() as *mut c_void, &mut err)};
    
    device.cl.borrow_mut().ptrs.push(r);

    if err != 0 {
        return Err(Error::from(OCLErrorKind::from_value(err)));
    }
    Ok(r)
}


#[cfg(feature="opencl")]
#[test]
fn test_unified_mem_bool() -> Result<(), Error> {
    let device = CLDevice::get(0)?;
    let um = device.unified_mem();
    println!("um: {um}");
    Ok(())
}

#[cfg(not(feature="safe"))]
#[cfg(feature="opencl")]
#[test]
fn test_unified_mem() -> Result<(), Error> {
    const TIMES: usize = 10000;
    use std::time::Instant;

    use custos::opencl::api::{create_buffer, MemFlags, release_mem_object, unified_ptr};

    let len = 20000;

    let data = vec![1f32; len];

    let device = CLDevice::get(0)?;
    
    if device.unified_mem() {
        let before = Instant::now();
        for _ in 0..TIMES {
            //std::thread::sleep(std::time::Duration::from_secs(1));
  
            let buf = create_buffer(&device.ctx(), MemFlags::MemReadWrite | MemFlags::MemUseHostPtr, len, Some(&data))?;
            
            let ptr = unified_ptr::<f32>(device.queue(), buf, len)?;

            let slice = unsafe {std::slice::from_raw_parts_mut(ptr, len)};

            for idx in 20..100 {
                slice[idx] = 4.;
            }

            unsafe { 
                release_mem_object(buf)?;
            }

            // 'data' vec is not freed
            assert_eq!(slice[25], 4.);

            /* 
            let mut read = vec![0f32; len];
            let event = unsafe { custos::opencl::api::enqueue_read_buffer(&device.queue(), buf, &mut read, true)}?;
            custos::opencl::api::wait_for_event(event)?;
            println!("read: {read:?}");
            */
        }


        let after = Instant::now();
        println!("use host ptr: {:?}", (after-before) / TIMES as u32);
            
        let before = Instant::now();
        for _ in 0..TIMES {        
            let buf = create_buffer(&device.ctx(), MemFlags::MemReadWrite | MemFlags::MemCopyHostPtr, len, Some(&data))?;
            let ptr = unified_ptr::<f32>(device.queue(), buf, len)?;
            let slice = unsafe {std::slice::from_raw_parts_mut(ptr, len)};
            
            for idx in 20..100 {
                slice[idx] = 4.;
            }
    
            unsafe { 
                release_mem_object(buf)?;
            }
        }
        let after = Instant::now();
        println!("copy host ptr: {:?}", (after-before) / TIMES as u32);
    
    }
    
    /* 
    let before = Instant::now();
    for _ in 0..TIMES {        
        let buf = create_buffer::<f32>(&device.ctx(), MemFlags::MemReadWrite as u64, len, None)?;
        
        unsafe { 
            release_mem_object(buf)?;
        }
    }

    let after = Instant::now();
    println!("alloc: {:?}", (after-before) / TIMES as u32);
    */
    
    Ok(())
}

#[cfg(not(feature="safe"))]
#[cfg(feature="opencl")]
#[test]
fn test_unified_calc() -> Result<(), Error> {

    let len = 100;
    
    let device = CPU::new();
    let mut a = Buffer::<f32>::new(&device, len);
    let mut b = Buffer::<f32>::from((&device, vec![1.; len]));

    let cl = CLDevice::get(0)?;
    
    let a: Buffer<f32> = Buffer {
        ptr: (null_mut(), unified_mem(&cl, a.as_mut_slice())?),
        len
    };
    let b = Buffer {
        ptr: (null_mut(), unified_mem(&cl, b.as_mut_slice())?),
        len,
    };

    cl_tew(&cl, &a, &b, "+")?;

//    let ptr = unified_ptr(cl.queue(), a)?;
//    let ptr = unified_ptr(cl.queue(), a)?;
    
    Ok(())
}

fn slice_add<T: Copy + std::ops::Add<Output = T>>(a: &[T], b: &[T], c: &mut [T]) {
    custos::cpu::element_wise_op_mut(a, b, c, |a, b| a+b)
}
#[test]
fn test_unified_mem_ops() -> Result<(), custos::Error> {
    let device = CLDevice::get(0)?;

    if device.unified_mem() {
        let a = Buffer::from((&device, [1, 4, 3, 2, 7, 9]));
        assert!(a.ptr.0 != std::ptr::null_mut());
         
        let b = Buffer::from((&device, [2, 1, 7, 4, 3, 2]));
        let mut c = Buffer::from((&device, [0; 6]));
        
        slice_add(&a, &b, &mut c);
    
        let res = device.read(&c);
        assert_eq!(res, c.as_slice().to_vec());
    }
    Ok(())
}

