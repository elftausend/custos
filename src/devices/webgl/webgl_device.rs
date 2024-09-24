use core::{cell::RefCell, ops::Deref};

use js_sys::wasm_bindgen::JsValue;
use std::rc::Rc;
use web_sys::{Element, WebGlFramebuffer, WebGlShader};

use crate::{
    webgl::error::WebGlError, wgsl::{WgslDevice, WgslShaderLaunch}, AddLayer, Alloc, Base, Buffer, CloneBuf, Device, Module, OnDropBuffer, OnNewBuffer, Read, RemoveLayer, Retrieve, Retriever, Setup, Shape, Unit, WrappedData, WriteBuf
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
    pub fn from_canvas<'a, NewMods>(
        maybe_canvas: Element,
    ) -> crate::Result<WebGL<SimpleMods::Module>>
    where
        SimpleMods: Module<'a, WebGL, Module = NewMods>,
        NewMods: Setup<WebGL<NewMods>>,
    {
        let mut webgl = WebGL {
            modules: SimpleMods::new(),
            device: WebGlDevice::new(maybe_canvas)?,
        };
        NewMods::setup(&mut webgl).unwrap();
        Ok(webgl)
    }

    pub fn new<'a, NewMods>() -> crate::Result<WebGL<SimpleMods::Module>>
    where
        SimpleMods: Module<'a, WebGL, Module = NewMods>,
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

impl WgslDevice for WebGL {
    #[inline]
    fn new(_idx: usize) -> crate::Result<Self> {
        WebGL::<Base>::new()
    }
}
impl Default for WebGL {
    #[inline]
    fn default() -> Self {
        WebGL::<Base>::new().unwrap()
    }
}

crate::impl_retriever!(WebGL, WebGlNumber + Copy + Default);
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
    type Base<T: Unit, S: crate::Shape> = WebGlData<T>;
    type Data<T: Unit, S: Shape> = Self::Wrap<T, Self::Base<T, S>>;

    type Error = JsValue;

    #[inline(always)]
    fn base_to_data<T: Unit, S: crate::Shape>(&self, base: Self::Base<T, S>) -> Self::Data<T, S> {
        self.wrap_in_base(base)
    }

    #[inline(always)]
    fn wrap_to_data<T: Unit, S: crate::Shape>(
        &self,
        wrap: Self::Wrap<T, Self::Base<T, S>>,
    ) -> Self::Data<T, S> {
        wrap
    }

    #[inline(always)]
    fn data_as_wrap<'a, T: Unit, S: crate::Shape>(
        data: &'a Self::Data<T, S>,
    ) -> &'a Self::Wrap<T, Self::Base<T, S>> {
        data
    }

    #[inline(always)]
    fn data_as_wrap_mut<'a, T: Unit, S: crate::Shape>(
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

impl<T, Mods, S> WriteBuf<T, S, Self> for WebGL<Mods>
where
    T: WebGlNumber + Default + Copy + 'static,
    Mods: OnDropBuffer,
    S: Shape,
{
    #[inline]
    fn write(&self, buf: &mut Buffer<T, Self, S>, data: &[T]) {
        buf.base_mut().write(data.iter().copied()).expect("Cannot write to buffer");
    }

    #[inline]
    fn write_buf(&self, dst: &mut Buffer<T, Self, S>, src: &Buffer<T, Self, S>) {
        // is there a way to do this without reading?
        self.write(dst, &src.read())
    }
}

impl<'a, T, Mods, S> CloneBuf<'a, T, S> for WebGL<Mods>
where
    T: WebGlNumber + Default + Copy + 'static,
    Mods: OnDropBuffer + OnNewBuffer<'a, T, Self, S>,
    S: Shape,
{
    #[inline]
    fn clone_buf(&'a self, buf: &Buffer<'a, T, Self, S>) -> Buffer<'a, T, Self, S> {
        let mut cloned = buf.empty_like();
        cloned.write(&buf.read());
        cloned
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

impl<Mods> WebGL<Mods> {
    #[inline]
    pub fn add_layer<Mod>(self) -> WebGL<Mod::Wrapped>
    where
        Mod: AddLayer<Mods, WebGL>,
    {
        WebGL {
            modules: Mod::wrap_layer(self.modules),
            device: self.device,
        }
    }

    pub fn remove_layer<NewMods>(self) -> WebGL<NewMods>
    where
        Mods: RemoveLayer<NewMods>,
    {
        WebGL {
            modules: self.modules.inner_mods(),
            device: self.device,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Base, WebGL};

    #[cfg(feature = "cached")]
    #[test]
    fn test_webgl_add_layer() {
        use crate::Cached;

        let device = WebGL::<Base>::new().unwrap();
        let device = device.add_layer::<Cached<()>>();
        let _device = device.remove_layer();
    }
}