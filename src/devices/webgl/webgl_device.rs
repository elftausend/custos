use core::{cell::RefCell, ops::Deref};

use js_sys::wasm_bindgen::JsValue;
use std::rc::Rc;
use web_sys::{Element, WebGlFramebuffer, WebGlShader};

use crate::{
    webgl::error::WebGlError, wgsl::WgslShaderLaunch, Alloc, Base, Buffer, Device, Module,
    OnDropBuffer, Read, Retrieve, Retriever, Setup, Shape, WrappedData,
};

use super::{
    context::Context,
    data::WebGlData,
    program::{AsWebGlShaderArgument, ProgramCache, WebGlNumber},
    vertex_attributes::VertexAttributes,
    vertex_shader,
};

pub struct WebGlDevice {
    pub context: Rc<Context>,
    pub vertex_attribs: VertexAttributes,
    pub frame_buf: WebGlFramebuffer,
    pub vertex_shader: WebGlShader,
    pub shader_cache: RefCell<ProgramCache>,
}

impl WebGlDevice {
    pub fn new(maybe_canvas: Element) -> crate::Result<WebGlDevice> {
        let context = Rc::new(Context::new(maybe_canvas).map_err(|_| WebGlError::ContextCreation)?);

        Ok(WebGlDevice {
            vertex_attribs: VertexAttributes::new(context.clone())?,
            vertex_shader: vertex_shader(&context)?,
            shader_cache: Default::default(),
            frame_buf: context
                .create_framebuffer()
                .ok_or(WebGlError::FrameBufferCreation)?,
            context,
        })
    }

    pub fn launch_shader<'a>(
        &'a self,
        src: impl AsRef<str>,
        gws: [u32; 3],
        args: &[&'a dyn AsWebGlShaderArgument],
    ) -> crate::Result<()> {
        let mut shader_cache = self.shader_cache.borrow_mut();

        let program = shader_cache.get(self.context.clone(), &self.vertex_shader, src)?;
        program.launch(&self.frame_buf, &self.vertex_attribs, args, gws)?;
        Ok(())
    }
}

impl Drop for WebGlDevice {
    #[inline]
    fn drop(&mut self) {
        self.context
            .delete_buffer(Some(self.vertex_attribs.position_buffer()));
        self.context
            .delete_buffer(Some(self.vertex_attribs.texcoords_buffer()));
    }
}

pub struct WebGL<Mods = Base> {
    pub modules: Mods,
    pub device: WebGlDevice,
}

impl<SimpleMods> WebGL<SimpleMods> {
    #[inline]
    pub fn from_canvas<NewMods>(maybe_canvas: Element) -> crate::Result<WebGL<SimpleMods::Module>>
    where
        SimpleMods: Module<WebGL, Module = NewMods>,
        NewMods: Setup<WebGL<NewMods>>,
    {
        let mut webgl = WebGL {
            modules: SimpleMods::new(),
            device: WebGlDevice::new(maybe_canvas)?,
        };
        NewMods::setup(&mut webgl).unwrap();
        Ok(webgl)
    }

    pub fn new<NewMods>() -> crate::Result<WebGL<SimpleMods::Module>>
    where
        SimpleMods: Module<WebGL, Module = NewMods>,
        NewMods: Setup<WebGL<NewMods>>,
    {
        let document = web_sys::window()
            .ok_or(WebGlError::MissingWindow)?
            .document()
            .ok_or(WebGlError::MissingDocument)?;
        let canvas = document
            .create_element("canvas")
            .map_err(|_| WebGlError::CanvasCreation)?;
        Ok(WebGL::<SimpleMods>::from_canvas(canvas).unwrap())
    }
}

crate::impl_buffer_hook_traits!(WebGL);
crate::impl_wrapped_data!(WebGL);

#[cfg(feature = "graph")]
crate::pass_down_optimize_mem_graph!(WebGL);

crate::pass_down_grad_fn!(WebGL);
crate::pass_down_tape_actions!(WebGL);

crate::pass_down_replace_buf_dev!(WebGL);
crate::pass_down_cursor!(WebGL);
crate::pass_down_cached_buffers!(WebGL);

impl<Mods: OnDropBuffer> Device for WebGL<Mods> {
    type Base<T, S: crate::Shape> = WebGlData<T>;
    type Data<T, S: Shape> = Self::Wrap<T, Self::Base<T, S>>;

    type Error = JsValue;

    #[inline(always)]
    fn base_to_data<T, S: crate::Shape>(&self, base: Self::Base<T, S>) -> Self::Data<T, S> {
        self.wrap_in_base(base)
    }

    #[inline(always)]
    fn wrap_to_data<T, S: crate::Shape>(
        &self,
        wrap: Self::Wrap<T, Self::Base<T, S>>,
    ) -> Self::Data<T, S> {
        wrap
    }

    #[inline(always)]
    fn data_as_wrap<'a, T, S: crate::Shape>(
        data: &'a Self::Data<T, S>,
    ) -> &'a Self::Wrap<T, Self::Base<T, S>> {
        data
    }

    #[inline(always)]
    fn data_as_wrap_mut<'a, T, S: crate::Shape>(
        data: &'a mut Self::Data<T, S>,
    ) -> &'a mut Self::Wrap<T, Self::Base<T, S>> {
        data
    }
}

impl<Mods> Deref for WebGL<Mods> {
    type Target = WebGlDevice;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl<T, Mods> Alloc<T> for WebGL<Mods>
where
    T: WebGlNumber + Default + Copy,
    Mods: OnDropBuffer,
{
    #[inline]
    fn alloc<S: Shape>(
        &self,
        len: usize,
        flag: crate::flag::AllocFlag,
    ) -> crate::Result<Self::Base<T, S>> {
        WebGlData::new(self.context.clone(), len, flag).ok_or(WebGlError::DataCreation.into())
    }

    fn alloc_from_slice<S: Shape>(&self, data: &[T]) -> crate::Result<Self::Base<T, S>> {
        let mut webgl_data = self.alloc::<S>(data.len(), crate::flag::AllocFlag::None)?;
        webgl_data.write(data.iter().copied());
        Ok(webgl_data)
    }
}

impl<T, Mods, S> Retriever<T, S> for WebGL<Mods>
where
    T: WebGlNumber + Default + Copy,
    Mods: Retrieve<Self, T, S>,
    S: Shape,
{
    fn retrieve<const NUM_PARENTS: usize>(
        &self,
        len: usize,
        parents: impl crate::Parents<NUM_PARENTS>,
    ) -> crate::Result<Buffer<T, Self, S>> {
        let data = unsafe { self.modules.retrieve::<NUM_PARENTS>(self, len, parents)? };
        let buf = Buffer {
            data,
            device: Some(self),
        };
        self.modules.on_retrieve_finish(&buf);
        Ok(buf)
    }
}

impl<T, Mods, S> Read<T, S> for WebGL<Mods>
where
    T: WebGlNumber + Default + Clone + 'static,
    Mods: OnDropBuffer,
    S: Shape,
{
    type Read<'a> = Vec<T>
    where
        Self: 'a,
        S: 'a;

    #[inline]
    fn read<'a>(&self, buf: &'a <Self as Device>::Base<T, S>) -> Self::Read<'a>
    where
        Self: 'a,
    {
        Read::<_, S>::read_to_vec(self, buf)
    }

    #[inline]
    fn read_to_vec(&self, buf: &<Self as Device>::Base<T, S>) -> Vec<T> {
        buf.read(&self.frame_buf, buf.out_idx.unwrap_or_default())
    }
}

impl<Mods> WgslShaderLaunch for WebGL<Mods> {
    type ShaderArg = dyn AsWebGlShaderArgument;

    #[inline]
    fn launch_shader(
        &self,
        src: impl AsRef<str>,
        gws: [u32; 3],
        args: &[&dyn AsWebGlShaderArgument],
    ) -> crate::Result<()> {
        self.device.launch_shader(src, gws, args)
    }
}
