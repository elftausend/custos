use custos::{prelude::*, CommonPtrs};

pub fn get_mut_slice<'a, T, D: Device>(buf: &'a mut Buffer<T, D>) -> &'a mut [T]
where
    D::Data<T, ()>: CommonPtrs<T>,
{
    unsafe { std::slice::from_raw_parts_mut(buf.ptrs_mut().0, buf.len()) }
}

pub fn get_slice<'a, T, D: Device>(buf: &'a Buffer<T, D>) -> &'a [T]
where
    D::Data<T, ()>: CommonPtrs<T>,
{
    unsafe { std::slice::from_raw_parts(buf.ptrs().0, buf.len()) }
}

#[cfg(feature = "std")]
pub fn read<T, D: Alloc<T>>(device: &D, buf: &Buffer<T, D>) -> Vec<T>
where
    D: Read<T> + Device,
    T: Clone + Default,
{
    device.read_to_vec(buf)
}

#[cfg(feature = "opencl")]
#[test]
fn test_cldevice_name() -> Result<(), Error> {
    let device = OpenCL::<Base>::new(chosen_cl_idx())?;
    println!("{}", device.name()?);

    Ok(())
}

#[cfg(feature = "opencl")]
#[test]
fn test_cldevice_version() -> Result<(), Error> {
    let device = OpenCL::<Base>::new(chosen_cl_idx())?;
    println!("{}", device.version()?);
    Ok(())
}

#[cfg(feature = "opencl")]
#[test]
fn test_cldevice_mem() -> Result<(), Error> {
    let device = OpenCL::<Base>::new(chosen_cl_idx())?;
    println!(
        "get_global_mem_size_in_gb: {}",
        device.global_mem_size_in_gb()?
    );
    println!("get_max_mem_alloc_in_gb: {}", device.max_mem_alloc_in_gb()?);
    Ok(())
}

#[cfg(feature = "opencl")]
#[test]
fn test_buffer_from_read() -> Result<(), Error> {
    let device = OpenCL::<Base>::new(chosen_cl_idx())?;

    let buf = Buffer::<f32, _>::from((&device, [3.13, 3., 1., 8.]));
    assert_eq!(read(&device, &buf), vec![3.13, 3., 1., 8.,]);

    let device = CPU::<Base>::new();

    let buf = Buffer::<f32, _>::from((&device, [3.13, 3., 1., 8.]));
    assert_eq!(read(&device, &buf), vec![3.13, 3., 1., 8.,]);
    Ok(())
}

#[cfg(feature = "opencl")]
#[test]
fn test_buffer_alloc_and_read() -> Result<(), Error> {
    let device = CPU::<Base>::new();

    let mut buf = Buffer::<u8, _>::new(&device, 10);

    let buf_slice = get_mut_slice(&mut buf);
    buf_slice.copy_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    assert_eq!(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], buf_slice);

    let cl = OpenCL::<Base>::new(chosen_cl_idx())?;

    let buf = Buffer::<f32, _>::from((&cl, [3.13, 3., 1., 8.]));
    let buf_read = read(&cl, &buf);
    assert_eq!(&[3.13, 3., 1., 8.], buf_read.as_slice());

    let buf = Buffer::<f32, _>::from((&device, [3.13, 3., 1., 8.]));
    let buf_read = read(&device, &buf);
    assert_eq!(&[3.13, 3., 1., 8.], buf_read.as_slice());

    let buf_read = get_slice(&buf);
    assert_eq!(&[3.13, 3., 1., 8.], buf_read);

    Ok(())
}

#[test]
fn test_buf_with_num() {
    let buf: Buffer<i32, ()> = 5.into();
    assert_eq!(buf.data.num, 5);

    let mut buf1: Buffer<_, ()> = 7f32.into();
    buf1.data.num = 3.;
    assert_eq!(buf1.data.num, 3.);
}

#[test]
fn test_use_number() {
    let num = Box::into_raw(Box::new(10));

    let num = unsafe { Box::from_raw(num) };
    assert_eq!(num, Box::new(10));
}

#[cfg(feature = "cpu")]
#[cfg(feature = "cached")]
#[cfg_attr(miri, ignore)]
#[test]
fn test_cached_cpu() {
    // for: cargo test -- --test-threads=1

    // std::env::set_var("RUST_BACKTRACE", "1");
    let device = CPU::<Cached<Base>>::new();

    let mut prev_ptr = None;

    for _ in device.range(0..100) {
        let buf: Buffer<f32, _> = device.retrieve::<0>(10, ()).unwrap();

        if prev_ptr.is_some() {
            assert_eq!(prev_ptr, Some(buf.data.ptr));
        }

        prev_ptr = Some(buf.data.ptr);
    }
}

/*#[test]
fn test_from_ptrs() {
    let mut value = 4f32;
    let ptr: *mut c_void = &mut value as *mut f32 as *mut c_void;

    let buf = Buffer::<f32>::from((ptr, 5));
    assert_eq!(buf.data.0, std::ptr::null_mut());

    let ptr: *mut f32 = &mut value as *mut f32;
    let buf = Buffer::<f32>::from((ptr, 5));
    assert_eq!(buf.data.1, std::ptr::null_mut());
}*/

#[cfg(feature = "cpu")]
#[test]
fn test_size_buf() {
    let x = core::mem::size_of::<Buffer<i8, CPU>>();
    println!("x: {x}");
}

fn _slice_add<T: Copy + std::ops::Add<Output = T>>(a: &[T], b: &[T], c: &mut [T]) {
    for i in 0..c.len() {
        c[i] = a[i] + b[i]
    }
}

use custos_macro::stack_cpu_test;

//#[cfg(feature = "cpu")]

#[stack_cpu_test]
#[test]
fn test_iterate_cpu() {
    let cmp = [1f32, 2., 3.3];

    let device = CPU::<Base>::new();
    let x = Buffer::with(&device, [1., 2., 3.3]);

    for (x, y) in x.iter().zip(cmp) {
        assert!(*x == y)
    }
}

#[cfg(feature = "opencl")]
#[test]
fn test_debug_print_buf() -> custos::Result<()> {
    let device = OpenCL::<Base>::new(chosen_cl_idx())?;

    let a = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));

    println!("a: {a:?}");
    Ok(())
}

#[cfg(unified_cl)]
fn slice_add<T, D>(_lhs: &Buffer<T, D>)
where
    D: Device,
    D::Data<T, ()>: std::ops::Deref<Target = [T]>,
{
}

#[cfg(unified_cl)]
#[test]
fn test_slice() {
    let device = CPU::<Base>::new();

    let buf = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));
    println!("buf: {:?}", buf.as_slice());

    let device = custos::OpenCL::<Base>::new(chosen_cl_idx()).unwrap();
    let buf = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));

    slice_add::<i32, _>(&buf);
}

#[cfg(feature = "cpu")]
#[test]
fn test_alloc() {
    let device = CPU::<Base>::new();

    let buf: Buffer<f32> = device.retrieve::<0>(100, ()).unwrap();
    assert_eq!(buf.read(), vec![0.; 100]);

    let buf: Buffer<f32> = device.retrieve::<0>(100, ()).unwrap();
    assert_eq!(buf.read(), vec![0.; 100]);
    drop(buf);
}

#[cfg(feature = "cpu")]
#[test]
fn test_deviceless_buf() {
    let mut buf = {
        let device = CPU::<Base>::new();
        Buffer::<u8, CPU>::deviceless(&device, 5)
    };

    println!("test buf ptr: {:?}", buf.ptrs());

    for (idx, element) in buf.iter_mut().enumerate() {
        *element = idx as u8;
    }
    assert_eq!(buf.as_slice(), &[0, 1, 2, 3, 4]);
}

/*
// compile-time error instead
#[test]
#[should_panic]
fn test_deviceless_buf_panic() {
    let buf = {
        let device = CPU::<Base>::new();
        Buffer::<u8>::deviceless(&device, 5)
    };
    buf.read();
}*/

/*
// compmile-time error
#[cfg(feature = "opencl")]
#[test]
fn test_deviceless_buf_cl() -> custos::Result<()> {
    use custos::WriteBuf;

    let buf = {
        let device = OpenCL::<Base>::new(chosen_cl_idx())?;
        let mut buf = Buffer::<u8, _>::deviceless(&device, 5);
        device.write(&mut buf, &[0, 1, 2, 3, 4]);
        drop(device);
        buf
    };

    let device = OpenCL::<Base>::new(chosen_cl_idx())?;
    assert_eq!(device.read(&buf), &[0, 1, 2, 3, 4]);

    Ok(())
}
*/

#[test]
fn test_buf_num() {
    let buf = Buffer::from(5);
    assert_eq!(**buf, 5);
}

#[cfg(feature = "cpu")]
#[test]
fn test_buf_const() {
    let _device = CPU::<Base>::new();
    // TODO
    //let device = Stack;
    // let buf = Buffer::with(&device, [1., 2., 3.]);
    // buf.read();

    //let buf = Buffer::from((&device, [1., 2., 3.]));
}
