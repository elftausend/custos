#[cfg(unified_cl)]
use custos::{Buffer, OpenCL};

#[cfg(unified_cl)]
fn main() -> custos::Result<()> {
    use custos::{opencl::chosen_cl_idx, Base};

    let device = OpenCL::<Base>::new(chosen_cl_idx())?;

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
    let cl_data = a.read_to_vec();
    assert_eq!(a.read(), cl_data);
    assert_eq!(&cl_data, &[3, 4, 5, 6, 7,]);

    Ok(())
}

#[cfg(not(unified_cl))]
fn main() {}
