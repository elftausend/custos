use std::borrow::Borrow;

use custos::{Buffer, Device, CacheAble2, Return, Alloc};



struct LinearLayer<'a, T: 'a, D: Device + 'a> {
    device: &'a D,
    inputs: Option<&'a Buffer<'a, T, D>>
}

pub trait AsRealRef<'a, T> {
    fn as_ref<'b>(&'a self) -> &'b T;
}

impl<'a, T, D: Device> AsRealRef<'a, Self> for &'a Buffer<'a, T, D> {
    fn as_ref<'b>(&'a self) -> &'b Self {
        self
    }
}

impl<'a, T, D: Device> LinearLayer<'a, T, D> {
    /*fn forward<In: AsRealRef<'a, Buffer<'a, T, D>> + 'a >(&mut self, inputs: In) -> Return<'a, T, D> 
    where
        D: Alloc<'a, T>,
    {
        let borrow: &'a Buffer<'a, T, D> = inputs.as_ref();
        self.inputs = Some(borrow);

        inputs.as_ref().device().retrieve(10, ())
    }*/

    fn forward2(&mut self, inputs: &'a Buffer<T, D>) -> Return<'a, T, D> 
    where
        D: Alloc<'a, T>,
    {
        self.inputs = Some(inputs);

        self.device.retrieve(10, ())
    }
}


#[cfg(test)]
mod tests {
    use custos::{CPU, Buffer, IsBuffer, Stack, Shape, stack::StackArray, Dim1};

    use super::LinearLayer;

    pub struct RefTest<'a> {
        text: &'a str,
    }

    impl<'a> RefTest<'a> {
        pub fn test<R: AsRef<str> + 'a>(&mut self, text: R) {
            //self.text = text.as_ref();            
        }    
    }
    
    #[test]
    fn test_test() {
        // test("text");
        // test("text".to_string());
    }


    #[test]
    fn test_linear_layer() {
        let device = CPU::new();
        let inputs = Buffer::from((&device, [1, 2, 3, 4, 5]));

        let mut lin1: LinearLayer<i32, CPU> = LinearLayer {
            device: &device,
            inputs: None
        };

        let mut lin2: LinearLayer<i32, CPU> = LinearLayer {
            device: &device,
            inputs: None
        };

        for _ in 0..100 {

            //let out = lin1.forward(&inputs);
            //let out = lin2.forward(out);

            let out = lin1.forward2(&inputs);

            // save for later for grads in linear layer is not nesecceray 
            // use impl or Return directly for forward input
            let out = lin2.forward2(out.buf());

            take_info(&Info {
                device: &Stack,
                array: StackArray::<Dim1<5>, f32>::new()    
            })
        }
        
    }

    pub fn take_info<T, S: Shape>(info: &Info<T, S>) {}

    pub struct Info<'a, T, S: Shape> {
        device: &'a Stack,
        array: StackArray<S, T>
    }
}

