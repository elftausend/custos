use custos::MTLDevice;

#[test]
fn test_device_mtl() {
    let device = MTLDevice::new();
    let _unified_mem = device.device.has_unified_memory();
    
}