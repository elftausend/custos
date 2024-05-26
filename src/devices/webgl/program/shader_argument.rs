use crate::{
    webgl::error::WebGlError,
    wgsl::{AsShaderArg, WgslShaderLaunch},
    Buffer, OnDropBuffer, Shape, WebGL,
};
use web_sys::{WebGl2RenderingContext, WebGlTexture, WebGlUniformLocation};

pub trait AsWebGlShaderArgument {
    #[inline]
    fn texture(&self) -> Option<&WebGlTexture> {
        None
    }
    #[inline]
    fn texture_width(&self) -> usize {
        0
    }
    #[inline]
    fn texture_height(&self) -> usize {
        0
    }
    #[inline]
    fn set_num_uniform(
        &self,
        _context: &WebGl2RenderingContext,
        _uniform_location: Option<&WebGlUniformLocation>,
        _type: &naga::Type,
    ) -> crate::Result<()> {
        Err(WebGlError::DatatypeArgumentMismatch.into())
    }
}

impl<'a, T: WebGlNumber, S: Shape, Mods: OnDropBuffer> AsWebGlShaderArgument
    for Buffer<'a, T, WebGL<Mods>, S>
{
    #[inline]
    fn texture(&self) -> Option<&WebGlTexture> {
        Some(&self.base().texture)
    }
    #[inline]
    fn texture_height(&self) -> usize {
        self.texture_height
    }
    #[inline]
    fn texture_width(&self) -> usize {
        self.texture_width
    }
}

pub trait WebGlNumber {
    const SCALAR_TYPE: naga::ScalarKind;

    fn set_num_uniform(
        &self,
        context: &WebGl2RenderingContext,
        uniform_location: Option<&WebGlUniformLocation>,
    );
}

impl WebGlNumber for f32 {
    const SCALAR_TYPE: naga::ScalarKind = naga::ScalarKind::Float;

    #[inline]
    fn set_num_uniform(
        &self,
        context: &WebGl2RenderingContext,
        uniform_location: Option<&WebGlUniformLocation>,
    ) {
        context.uniform1f(uniform_location, *self)
    }
}

impl WebGlNumber for i32 {
    const SCALAR_TYPE: naga::ScalarKind = naga::ScalarKind::Sint;

    #[inline]
    fn set_num_uniform(
        &self,
        context: &WebGl2RenderingContext,
        uniform_location: Option<&WebGlUniformLocation>,
    ) {
        context.uniform1i(uniform_location, *self)
    }
}
impl WebGlNumber for u32 {
    const SCALAR_TYPE: naga::ScalarKind = naga::ScalarKind::Uint;

    #[inline]
    fn set_num_uniform(
        &self,
        context: &WebGl2RenderingContext,
        uniform_location: Option<&WebGlUniformLocation>,
    ) {
        context.uniform1ui(uniform_location, *self)
    }
}

impl<T: WebGlNumber> AsWebGlShaderArgument for T {
    #[inline]
    fn set_num_uniform(
        &self,
        context: &WebGl2RenderingContext,
        uniform_location: Option<&WebGlUniformLocation>,
        ty: &naga::Type,
    ) -> crate::Result<()> {
        let naga::TypeInner::Scalar(naga::Scalar { kind, width: _ }) = ty.inner else {
            return Err(WebGlError::DatatypeArgumentMismatch.into());
        };
        if kind != T::SCALAR_TYPE {
            return Err(WebGlError::DatatypeArgumentMismatch.into());
        }
        Ok(self.set_num_uniform(context, uniform_location))
    }
}

impl<T: AsWebGlShaderArgument + 'static> AsShaderArg<WebGL> for T {
    #[inline]
    fn arg(&self) -> &<WebGL as WgslShaderLaunch>::ShaderArg {
        self
    }

    #[inline]
    fn arg_mut(&mut self) -> &mut <WebGL as WgslShaderLaunch>::ShaderArg {
        self
    }
}
