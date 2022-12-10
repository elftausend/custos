use custos::prelude::*;
use cuwanto_server as cuw;

#[test]
fn test_network_device() -> cuw::Result<()> {
    let device = Network::new("127.0.0.1:11001", cuw::DeviceType::CPU)?;
    //let device = CPU::new();
    //let device = Stack::new().unwrap();
    //let buf = device.cuw_client.borrow_mut().alloc_buf::<f64>(4)?;
    let buf = Buffer::<f64, _>::from((&device, &[1., 2., 3., 4.]));
    // ....
    println!("buf: {:?}", buf.read());
    //buf.clear();
    Ok(())
}
