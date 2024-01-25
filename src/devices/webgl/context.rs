use core::ops::{Deref, DerefMut};

use js_sys::wasm_bindgen::{JsCast, JsValue};
use web_sys::Element;

pub struct Context {
    pub context: web_sys::WebGl2RenderingContext,
}

impl Context {
    pub fn new(maybe_canvas: Element) -> Result<Self, JsValue> {
        let canvas = maybe_canvas.dyn_into::<web_sys::HtmlCanvasElement>()?;

        let context = canvas
            .get_context("webgl2")?
            .unwrap()
            .dyn_into::<web_sys::WebGl2RenderingContext>()?;

        Ok(Self { context })
    }
}

impl Deref for Context {
    type Target = web_sys::WebGl2RenderingContext;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.context
    }
}

impl DerefMut for Context {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.context
    }
}
