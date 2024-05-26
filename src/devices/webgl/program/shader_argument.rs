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
    const INTERNAL_FORMAT: u32;
    const FORMAT: u32;
    const TYPE: u32;

    fn set_num_uniform(
        &self,
        context: &WebGl2RenderingContext,
        uniform_location: Option<&WebGlUniformLocation>,
    );

    unsafe fn array_view(upload_data: &[Self]) -> js_sys::Object 
    where 
        Self: Sized 
    {
        let texture_data = unsafe {
            std::slice::from_raw_parts(upload_data.as_ptr() as *const u8, upload_data.len() * 4)
        };
        js_sys::Uint8Array::view(texture_data).into()
    }
    fn read_pixels(context: &WebGl2RenderingContext, texture_width: usize, texture_height: usize) -> Vec<Self>
    where
        Self: Sized;
}

impl WebGlNumber for f32 {
    const SCALAR_TYPE: naga::ScalarKind = naga::ScalarKind::Float;
    const INTERNAL_FORMAT: u32 = WebGl2RenderingContext::RGBA8;
    const FORMAT: u32 = WebGl2RenderingContext::RGBA;
    const TYPE: u32 = WebGl2RenderingContext::UNSIGNED_BYTE;

    #[inline]
    fn set_num_uniform(
        &self,
        context: &WebGl2RenderingContext,
        uniform_location: Option<&WebGlUniformLocation>,
    ) {
        context.uniform1f(uniform_location, *self)
    }

    unsafe fn array_view(upload_data: &[Self]) -> js_sys::Object {
        let texture_data = unsafe {
            std::slice::from_raw_parts(upload_data.as_ptr() as *const u8, upload_data.len() * 4)
        };
        js_sys::Uint8Array::view(texture_data).into()
    }
    
    fn read_pixels(context: &WebGl2RenderingContext, texture_width: usize, texture_height: usize) -> Vec<Self> {
        let mut read_data = vec![Self::default(); texture_height * texture_width];
        let texture_data = unsafe {
            std::slice::from_raw_parts_mut(read_data.as_mut_ptr() as *mut u8, read_data.len() * 4)
        };

        context
            .read_pixels_with_u8_array_and_dst_offset(
                0,
                0,
                texture_width as i32,
                texture_height as i32,
                WebGl2RenderingContext::RGBA,
                WebGl2RenderingContext::UNSIGNED_BYTE,
                texture_data,
                0,
            )
            .unwrap();

        read_data
    } 
}

impl WebGlNumber for i32 {
    const SCALAR_TYPE: naga::ScalarKind = naga::ScalarKind::Sint;
    const INTERNAL_FORMAT: u32 = WebGl2RenderingContext::RGBA8I;
    const FORMAT: u32 = WebGl2RenderingContext::RGBA_INTEGER;
    const TYPE: u32 = WebGl2RenderingContext::BYTE;

    #[inline]
    fn set_num_uniform(
        &self,
        context: &WebGl2RenderingContext,
        uniform_location: Option<&WebGlUniformLocation>,
    ) {
        context.uniform1i(uniform_location, *self)
    }
    

    unsafe fn array_view(upload_data: &[Self]) -> js_sys::Object {
        let texture_data = unsafe {
            std::slice::from_raw_parts(upload_data.as_ptr() as *const i8, upload_data.len() * 4)
        };
        js_sys::Int8Array::view(texture_data).into()
    }
    
    fn read_pixels(context: &WebGl2RenderingContext, texture_width: usize, texture_height: usize) -> Vec<Self> {
        todo!()
    }
    
}
impl WebGlNumber for u32 {
    const SCALAR_TYPE: naga::ScalarKind = naga::ScalarKind::Uint;
    const INTERNAL_FORMAT: u32 = WebGl2RenderingContext::RGBA8UI;
    const FORMAT: u32 = WebGl2RenderingContext::RGBA_INTEGER;
    const TYPE: u32 = WebGl2RenderingContext::UNSIGNED_BYTE;

    #[inline]
    fn set_num_uniform(
        &self,
        context: &WebGl2RenderingContext,
        uniform_location: Option<&WebGlUniformLocation>,
    ) {
        context.uniform1ui(uniform_location, *self)
    }
   
    fn read_pixels(context: &WebGl2RenderingContext, texture_width: usize, texture_height: usize) -> Vec<Self>
    where
        Self: Sized 
    {
        let mut read_data = vec![Self::default(); texture_height * texture_width * 4];

        { 
            let texture_data = unsafe { js_sys::Uint32Array::view(&mut read_data) };
            context
                .read_pixels_with_array_buffer_view_and_dst_offset(
                    0,
                    0,
                    texture_width as i32,
                    texture_height as i32,
                    WebGl2RenderingContext::RGBA_INTEGER,
                    WebGl2RenderingContext::UNSIGNED_INT,
                    &texture_data,
                    0,
                ).unwrap();
        }
        let read_data = read_data.iter().map(|&x| x as u8).collect::<Vec<_>>();
        read_data.chunks(4).map(|x| u32::from_ne_bytes([x[0], x[1], x[2], x[3]])).collect()
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
