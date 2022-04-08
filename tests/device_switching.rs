#[cfg(feature="opencl")]
use custos::{opencl::CLDevice, AsDev, Matrix, range, cpu::CPU, BaseOps};

#[cfg(feature="opencl")]
#[test]
fn test_device_switching() -> Result<(), custos::Error> {
    let device = CLDevice::get(0)?.select();
    let a = Matrix::from(( &device, (2, 3), [1.51f32, 6.123, 7., 5.21, 8.62, 4.765]));
    let b = Matrix::from(( &device, (2, 3), [1.51f32, 6.123, 7., 5.21, 8.62, 4.765]));
    
    for _ in range(500) {
        let c = &a + &b;
    
        let cpu = CPU::new();
        let c = Matrix::from( (&cpu, c.dims(), c.read()) );
        let d_cpu = cpu.add(&c, &c);
    
        let d = Matrix::from( (&device, d_cpu) );
        assert_eq!(vec![6.04, 24.492, 28., 20.84, 34.48, 19.06], d.read());   
    }
    Ok(())
}
