use custos::{Buffer, OpenCL, VecRead};

#[cfg(unified_cl)]
fn main() -> custos::Result<()> {
    let device = OpenCL::new(0)?;

    // declare function with conditional compilation attribute #[cfg(unified_cl)] or check dynamically:
    // if !device.unified_mem() {
    //     println!("CLDevice uses own memory");
    //     return Ok(());
    // }

    // create an OpenCL buffer
    let mut a = Buffer::from((&device, [1, 2, 3, 4, 5]));

    // This OpenCL buffer is used to iterate all values with a for loop,
    // which only works on devices with unified memory.
    for value in &mut a {
        *value += 2;
    }

    // Read OpenCL buffer.
    // This yields the same data as the corresponding CPU slice.
    let cl_data = device.read(&a);
    assert_eq!(a.as_slice(), &cl_data);
    assert_eq!(&cl_data, &[3, 4, 5, 6, 7,]);

    Ok(())
}

#[cfg(not(unified_cl))]
fn main() {}