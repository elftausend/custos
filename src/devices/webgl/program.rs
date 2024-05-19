use crate::{webgl::error::WebGlError, wgsl::Glsl};
use core::ops::Deref;
use naga::back::glsl::ReflectionInfoCompute;
use std::rc::Rc;
use web_sys::{WebGl2RenderingContext, WebGlProgram, WebGlShader};

use super::context::Context;

mod cache;
pub use cache::*;
mod launch;
mod shader_argument;
pub use shader_argument::*;

#[derive(Debug)]
pub struct Program {
    program: WebGlProgram,
    frag_shader: WebGlShader,
    pub reflection_info: ReflectionInfoCompute,
    pub module: naga::Module,
    context: std::rc::Rc<Context>,
}

impl Deref for Program {
    type Target = WebGlProgram;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.program
    }
}

impl Program {
    pub fn new(
        context: Rc<Context>,
        vert_shader: &WebGlShader,
        src: &impl AsRef<str>,
    ) -> crate::Result<Self> {
        let mut glsl = Glsl::from_wgsl_compute(src)?;
        let shader_source = glsl.sources.1.pop().ok_or(WebGlError::MissingShader)?;
        let frag_shader = shader_source.compile(&context)?;
        let program = link_program(&context, vert_shader, &frag_shader)?;

        Ok(Program {
            program,
            frag_shader,
            context,
            module: glsl.sources.0,
            reflection_info: shader_source.reflection_info.unwrap(),
        })
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        self.context.delete_program(Some(&self.program));
        self.context.delete_shader(Some(&self.frag_shader))
    }
}

pub fn link_program(
    context: &WebGl2RenderingContext,
    vert_shader: &WebGlShader,
    frag_shader: &WebGlShader,
) -> Result<WebGlProgram, String> {
    let program = context
        .create_program()
        .ok_or_else(|| String::from("Unable to create shader object"))?;

    context.attach_shader(&program, vert_shader);
    context.attach_shader(&program, frag_shader);
    context.link_program(&program);

    if context
        .get_program_parameter(&program, WebGl2RenderingContext::LINK_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(program)
    } else {
        Err(context
            .get_program_info_log(&program)
            .unwrap_or_else(|| String::from("Unknown error creating program object")))
    }
}
