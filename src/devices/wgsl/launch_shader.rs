pub trait WgslShaderLaunch<S: ?Sized> {
    fn launch_shader(&self, src: impl AsRef<str>, gws: [u32; 3], args: &[&S]) -> crate::Result<()>;
}
