use custos::{Matrix, CPU, AsDev, cpu::CPU_CACHE, Node};

fn main() {
    let device = CPU::new().select();

    let a = Matrix::<i16>::new(&device, (100, 100));
    let b = Matrix::<i16>::new(&device, (100, 100));

    let out = a + b;
    let info = CPU_CACHE.with(|cache| {
        let cache = cache.borrow();
        let mut node = Node::new(100*100);
        node.idx = 0;
        *cache.nodes.get(&node).unwrap()
     });
     assert!(info.0.0 == out.data().ptr.0 as *mut usize);
}