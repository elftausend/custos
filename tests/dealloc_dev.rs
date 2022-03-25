use custos::{libs::cpu::CPU2, AsDev2, Matrix};


#[test]
fn test_rc_get_dev() {
    CPU2::new().select();

    let a = Matrix::from( ((2, 3), &[1., 2., 3., 4., 5., 6.,]) );
    let b = Matrix::from( ((2, 3), &[6., 5., 4., 3., 2., 1.,]) );

    let c = a+b;   
    
}