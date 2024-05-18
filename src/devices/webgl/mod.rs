mod context;
mod data;
mod error;
mod program;
mod vertex_attributes;
mod webgl_device;

use web_sys::{WebGl2RenderingContext, WebGlShader};
pub use webgl_device::*;

use crate::wgsl::compile_shader;

pub fn vertex_shader(context: &WebGl2RenderingContext) -> crate::Result<WebGlShader> {
    Ok(compile_shader(
        context,
        WebGl2RenderingContext::VERTEX_SHADER,
        r##"#version 300 es
        precision highp float;

        in vec4 position;
        in vec2 texcoords;
        out vec2 thread_uv;

        void main() {
            thread_uv = texcoords;
            gl_Position = position;
        }
        "##,
    )?)
}
