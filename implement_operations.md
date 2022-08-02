# custos - implement operations for all compute devices (CPU, OpenCL, CUDA).

As code in .md files is not compiled directly, minor compilation issues may appear in the snippets below. 
Hence, you can find the entire source code in the examples folder, or just click [here](https://github.com/elftausend/custos/blob/main/examples/implement_operations.rs).

## Make operations invocable für all devices

Implementing a new operation happens in a few steps. <br>

The first step is to define a trait. This is the operation<br>
This trait is then implemented for all devices.<br>

You can use your own type as parameters also. It just needs to encapsulate a ```Buffer```.

```rust
/// AddBuf will be implemented for all compute devices.
pub trait AddBuf<T> {
    /// This operation perfoms element-wise addition.
    fn add(&self, lhs: &Buffer<T>, rhs: &Buffer<T>) -> Buffer<T>;
    // ... you can add more operations if you want to do that.
}
```

Afterwards, implement this trait for every device.<br>
However, it is not mandatory to implement this trait for every device, but we will come to this again a bit later.

```rust
// Host CPU implementation
impl<T> AddBuf<T> for CPU 
where
    T: Copy + Default + std::ops::Add<Output=T> // instead of adding a lot of trait bounds, 
{                                               // you can use the custos::Number trait. This trait is implemented for all number types (usize, i16, f32, ...)
    fn add(&self, lhs: &Buffer<T>, rhs: &Buffer<T>) -> Buffer<T> {
        let len = std::cmp::min(lhs.len, rhs.len);

        // this returns a previously allocated buffer. 
        // You can deactivate the caching behaviour by adding the "realloc" feature 
        // to the custos feature list in the Cargo.toml.
        let mut out = Cache::get(self, len);

        // By default, the Buffer dereferences to a slice. 
        // Therefore, standard indexing can be used. 
        // You can pass a CPU Buffer to a function that takes a slice as a parameter, too.
        // However, the Buffer must be created via a CPU.
        for i in 0..len {
            out[i] = lhs[i] + rhs[i];
        }
        out
    }
}

// OpenCL implementation
impl<T> AddBuf<T> for CLDevice 
where
    T: CDatatype // the custos::CDatatype trait is used to 
{                // get the OpenCL C type string for creating generic OpenCL kernels.
    fn add(&self, lhs: &Buffer<T>, rhs: &Buffer<T>) -> Buffer<T> {

        // generic OpenCL kernel
        let src = format!("
            __kernel void add(__global const {datatype}* lhs, __global const {datatype}* rhs, __global {datatype}* out) 
            {{
                size_t id = get_global_id(0);
                out[id] = lhs[id] + rhs[id];
            }}
        ", datatype=T::as_c_type_str());

        let len = std::cmp::min(lhs.len, rhs.len);
        let out = Cache::get::<T, CLDevice>(self, len);

        // In the background, the kernel is compiled once. After that, it will be reused every iteration.
        // The cached kernels are released (or freed) when the underlying CLDevice is dropped.
        // The arguments are specified with a slice of buffers and/or numbers.
        enqueue_kernel(self, &src, [len, 0, 0], None, &[&lhs, &rhs, &out]).unwrap();
        out
    }
}

// CUDA Implementation
impl<T: CDatatype> AddBuf<T> for CudaDevice {
    fn add(&self, lhs: &Buffer<T>, rhs: &Buffer<T>) -> Buffer<T> {

        // generic CUDA kernel
        let src = format!(
            r#"extern "C" __global__ void add({datatype}* lhs, {datatype}* rhs, {datatype}* out, int numElements)
                {{
                    int idx = blockDim.x * blockIdx.x + threadIdx.x;
                    if (idx < numElements) {{
                        out[idx] = lhs[idx] + rhs[idx];
                    }}
                    
                }}
        "#, datatype = T::as_c_type_str());
        
        let len = std::cmp::min(lhs.len, rhs.len);
        let out = Cache::get::<T, CudaDevice>(self, len);

        // The kernel is compiled once with nvrtc and is cached too.
        // The arguments are specified with a vector of buffers and/or numbers.
        launch_kernel1d(len, self, &src, "add", vec![lhs, rhs, &out, &len]).unwrap();
        out
    
    }
}
```
Now, the operation is accessible via the devices.<br>
Let's try it out with a ```CPU``` device:

```rust
let device = CPU::new();

let lhs = Buffer::from((&device, [1, 3, 5, 3, 2, 6]));
let rhs = Buffer::from((&device, [-1, -12, -6, 3, 2, -1]));

let out = device.add(&lhs, &rhs);
assert_eq!(out.read(), vec![0, -9, -1, 6, 4, 5]); // to read a CPU Buffer, you can also call .as_slice() on it.
```

If you want to use another device, just update the device declaration.

```rust
fn main() -> custos::Result<()> {
    let device = CLDevice::new(0)?; 
    // or:
    let device = CudaDevice::new(0)?;
    Ok(())
}
```

## Make operations invocable on ```Buffer``` or custom structs

Now, we have implemented a custom operation for all compute devices. 
However, you may have spot something.

```rust
let out = device.add(&lhs, &rhs);
assert_eq!(out.read(), vec![0, -9, -1, 6, 4, 5]);
```

If you take a closer look at both statements, you may notice that the ```read()``` function is invoked on a buffer without specifying any device, even though reading a buffer is device specific.
We can implement this for ```AddBuf``` as well.<br>
To get this to work, a new trait must be created. If the operation should be used on a struct created in your current crate, you can omit this step.

```rust
pub trait AddOp<T> {
    fn add(&self, rhs: &Buffer<T>) -> Buffer<T>;
}
```

This trait is then implemented for ```Buffer``` (or any other type).

```rust
impl<T: CDatatype> AddOp<T> for Buffer<'_, T> {
    #[inline]
    fn add(&self, rhs: &Buffer<T>) -> Buffer<T> {
        get_device!(self.device, AddBuf<T>).add(self, rhs)
    }
}
```

If you have defined your own struct that encapsulates a ```Buffer```, you can do the following:

```rust
pub struct OwnStruct<'a, T> {
    buf: Buffer<'a, T>
}

impl<'a, T> OwnStruct<'a, T> {
    #[inline]
    fn add(&self, rhs: &OwnStruct<T>) -> Buffer<T> 
    where 
        T: CDatatype
    {
        get_device!(self.buf.device, AddBuf<T>).add(&self.buf, &rhs.buf)
    }

    // general context
    #[inline]
    fn operation(&self, rhs: &OwnStruct<T>, other_arg: &T) -> OwnStruct<T> {
        get_device!(self.buf.device, OperationTrait<T>).operation(self, rhs, other_arg)
    }

    // ... more operations ... 
}
```

### Issues with ```get_device!```

As mentioned before, it is not mandatory to implement an operation for every device.<br>
However, there is one caveat. <br>
If you want to call new operations on the ```Buffer``` (or any custom type), a ```CPU``` implementation must be available, since ```get_device!``` expects this. <br>

A way around this, if you only want this functionality, for instance, for OpenCL buffers, may be a trait where all operations are ```unimplemented!()``` by default.
For example:

```rust
// this trait is implemented for all devices.
pub trait AnotherOpBuf<T> {
    fn operation(&self, _buf: Buffer<T>) -> Buffer<T>{
        unimplemented!()
    }
}
```

Then, this trait can be implemented for ```CPU```, without providing any functionality.
```rust
impl<T> AnotherOpBuf<T> for CPU {}
```

For ```CLDevice```, the operation can be properly implemented.

```rust
impl<T> AnotherOpBuf<T> for CLDevice {
    #[inline]
    fn operation(&self, _buf: Buffer<T>) -> Buffer<T> {
        // ...
        todo!()
    }
}
```

Currently, If you use ```get_device!``` in your crate, you also need to add a 'cuda' and a 'opencl' feature to your Cargo.toml.

```toml
[features]
default = ["opencl"]
opencl = []
cuda = []
```

This is because ```get_device!``` contains code that uses ```#[cfg(feature="cuda")]``` and ```#[cfg(feature="opencl")]```. This macro is then expanded in your crate. Therefore, the expanded code looks after these features, which would fail without adding these features.