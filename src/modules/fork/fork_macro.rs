#[macro_export]
macro_rules! fork {
    // use another macro
    ($device: ident, $cpu_op:expr, $gpu_op:expr, $input_lengths:expr) => {
        #[cfg(unified_cl)] // FIXME: this is expanded in user's code -> flag is probably not active
        {
            $device.use_cpu_or_gpu((file!(), line!(), column!()).into(), &[], $cpu_op, $gpu_op);
            return;
        }
    };
}
