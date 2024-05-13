use core::ops::Deref;

use js_sys::wasm_bindgen::JsValue;
use std::rc::Rc;
use web_sys::Element;

use crate::{
    webgl::error::WebGlError, Alloc, Base, Buffer, Device, Module, OnDropBuffer, Retrieve,
    Retriever, Setup, Shape, WrappedData,
};

use super::{context::Context, data::WebGlData};

pub struct WebGL<Mods = Base> {
    pub modules: Mods,
    pub context: Rc<Context>,
}

impl<SimpleMods> WebGL<SimpleMods> {
    #[inline]
    pub fn from_canvas<NewMods>(maybe_canvas: Element) -> Result<WebGL<SimpleMods::Module>, JsValue>
    where
        SimpleMods: Module<WebGL, Module = NewMods>,
        NewMods: Setup<WebGL<NewMods>>,
    {
        let mut webgl = WebGL {
            modules: SimpleMods::new(),
            context: Rc::new(Context::new(maybe_canvas)?),
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
    type Target = Context;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.context
    }
}

impl<Mods: OnDropBuffer> Alloc<f32> for WebGL<Mods> {
    #[inline]
    fn alloc<S: Shape>(
        &self,
        len: usize,
        flag: crate::flag::AllocFlag,
    ) -> crate::Result<Self::Base<f32, S>> {
        WebGlData::new(self.context.clone(), len, flag).ok_or(WebGlError::DataCreation.into())
    }

    fn alloc_from_slice<S: Shape>(&self, data: &[f32]) -> crate::Result<Self::Base<f32, S>> {
        let mut webgl_data = self.alloc::<S>(data.len(), crate::flag::AllocFlag::None)?;
        webgl_data.write(data);
        Ok(webgl_data)
    }
}

impl<Mods: Retrieve<Self, f32, S>, S: Shape> Retriever<f32, S> for WebGL<Mods> {
    fn retrieve<const NUM_PARENTS: usize>(
        &self,
        len: usize,
        parents: impl crate::Parents<NUM_PARENTS>,
    ) -> crate::Result<Buffer<f32, Self, S>> {
        let data = unsafe { self.modules.retrieve::<NUM_PARENTS>(self, len, parents)? };
        let buf = Buffer {
            data,
            device: Some(self),
        };
        self.modules.on_retrieve_finish(&buf);
        Ok(buf)
    }
}
