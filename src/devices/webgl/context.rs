use core::{
    cell::Cell,
    ops::{Deref, DerefMut},
};

use js_sys::wasm_bindgen::{JsCast, JsValue};
use web_sys::Element;

#[derive(Debug)]
pub struct Context {
    pub context: web_sys::WebGl2RenderingContext,
    pub highest_id: Cell<usize>,
}

impl Context {
    pub fn new(maybe_canvas: Element) -> Result<Self, JsValue> {
        let canvas = maybe_canvas.dyn_into::<web_sys::HtmlCanvasElement>()?;

        let context = canvas
            .get_context("webgl2")?
            .expect("Could not get webgl2 context")
            .dyn_into::<web_sys::WebGl2RenderingContext>()?;

        Ok(Self {
            context,
            highest_id: Default::default(),
        })
    }

    pub fn gen_id(&self) -> usize {
        let id = self.highest_id.get();
        self.highest_id.set(id + 1);
        id + 1
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
