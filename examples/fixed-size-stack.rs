use custos::prelude::*;

fn main() {
    let stack = Stack::new();
    let _buf = Buffer::<i32, _, Dim1<3>>::from((&stack, [1, 2, 3]));

    let buf_stack = Buffer::with(&stack, [1, 2, 3]);

    let cpu = CPU::<Base>::new();
    let buf_heap = Buffer::with(&cpu, [1, 2, 3]);

    assert_eq!(buf_stack.read(), buf_heap.read());

    let buf_stack_dim2 = Buffer::with(&stack, [[1, 2, 3], [7, 2, 1]]);
    let buf_heap_dim2 = Buffer::with(&cpu, [[1, 2, 3], [7, 2, 1]]);

    assert_eq!(buf_heap_dim2.read_to_vec(), buf_stack_dim2.read_to_vec());
}
