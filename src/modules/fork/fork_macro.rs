#[macro_export]
macro_rules! fork {
    ($device: ident, $cpu_op:expr, $gpu_op:expr, $input_lengths:expr) => {
        if cfg!(unified_cl) {
            $device.use_cpu_or_gpu((file!(), line!(), column!()).into(), &[], $cpu_op, $gpu_op);
            return;
        }
    };
}
