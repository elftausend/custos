use std::borrow::Borrow;

use custos::{Buffer, Device, CacheAble2, Return, Alloc};



struct LinearLayer<'a, T: 'a, D: Device + 'a> {
    inputs: Option<Return<'a, T, D>>
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

    fn forward2(&mut self, inputs: Return<'a, T, D>) -> Return<'a, T, D> 
    where
        D: Alloc<'a, T>,
    {
        self.inputs = Some(inputs);

        inputs.device().retrieve(10, ())
    }
}


#[cfg(test)]
mod tests {
    use custos::{CPU, Buffer};

    use super::LinearLayer;

    pub struct RefTest<'a> {
        text: &'a str,
    }

    impl<'a> RefTest<'a> {
        pub fn test<R: AsRef<str> + 'a>(&mut self, text: R) {
            self.text = text.as_ref();            
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
            inputs: None
        };

        let mut lin2: LinearLayer<i32, CPU> = LinearLayer {
            inputs: None
        };

        for _ in 0..100 {

            //let out = lin1.forward(&inputs);
            //let out = lin2.forward(out);

            let out = lin1.forward2(&inputs);
            let out = lin2.forward2(out);
        }
        
    }
}