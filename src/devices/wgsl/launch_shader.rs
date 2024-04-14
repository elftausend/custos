pub trait WgslShaderLaunch {
    type ShaderArg: ?Sized;
    fn launch_shader(
        &self,
        src: impl AsRef<str>,
        gws: [u32; 3],
        args: &[&Self::ShaderArg],
    ) -> crate::Result<()>;
}

pub trait AsShaderArg<D: WgslShaderLaunch> {
    fn arg(&self) -> &D::ShaderArg;
    fn arg_mut(&mut self) -> &mut D::ShaderArg;
}
