#[cfg(test)]
mod tests {
    use crate::{
        derivative, AddOperation, Autograd, Base, Combiner, Device, OpenCL, Resolve,
        UnaryElementWiseMayGrad, CPU,
    };

    #[test]
    fn test_ew_fuse() {
        let device = OpenCL::<Autograd<Base>>::new(0).unwrap();
        // let device = CPU::<Autograd<Base>>::new();

        // let x = device.buffer([1, 2, 3]);
        // let mut a = x.empty_like();
        // let y = device.buffer([1, 2, 3]);
        // let z = device.buffer([1, 2, 3]);
        // let mut b = a.empty_like();

        // // (x + y) * z

        // device
        //     .add_op((&x, &y, &mut a), |(x, y, a)| {
        //         for ((x, y), a) in x.iter().zip(y.iter()).zip(a.iter_mut()) {
        //             *a = x + y;
        //         }
        //         Ok(())
        //     })
        //     .unwrap();

        // device
        //     .add_op((&a, &z, &mut b), |(a, z, b)| {
        //         for ((a, z), b) in a.iter().zip(z.iter()).zip(b.iter_mut()) {
        //             *b = a * z;
        //         }
        //         Ok(())
        //     })
        //     .unwrap();

        let x = device.buffer([1., 2., 3.]);

        let forward_fn = |x: Resolve<f32>| x.sin().add(3.);
        let out = device.unary_ew(&x, forward_fn, move |x| derivative!(forward_fn, x));

        out.backward();
        let grad = x.grad();
        assert_eq!(grad.read(), &[1f32.cos(), 2f32.cos(), 3f32.cos()]);
        println!("grad: {grad:?}");
    }
}
