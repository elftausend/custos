use js_sys::wasm_bindgen::{JsCast, JsValue};
use web_sys::Element;

pub struct Context {
    pub(crate) context: web_sys::WebGl2RenderingContext,
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
