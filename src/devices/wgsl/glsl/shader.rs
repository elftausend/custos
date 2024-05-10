use naga::{back::glsl::ReflectionInfoCompute, ShaderStage};
use web_sys::{WebGl2RenderingContext, WebGlShader};

use super::GlslError;

#[derive(Debug, Clone)]
pub struct ShaderSource {
    pub shader_stage: ShaderStage,
    pub src: String,
    pub reflection_info: Option<ReflectionInfoCompute>,
}

impl ShaderSource {
    pub fn compile(&self, context: &WebGl2RenderingContext) -> Result<WebGlShader, GlslError> {
        let shader_type = match self.shader_stage {
            ShaderStage::Vertex => WebGl2RenderingContext::VERTEX_SHADER,
            ShaderStage::Fragment => WebGl2RenderingContext::FRAGMENT_SHADER,
            _ => panic!("Unsupported shader stage: {:?}", self.shader_stage),
        };
        compile_shader(context, shader_type, &self.src)
    }
}

pub fn compile_shader(
    context: &WebGl2RenderingContext,
    shader_type: u32,
    source: &str,
) -> Result<WebGlShader, GlslError> {
    let shader = context
        .create_shader(shader_type)
        .ok_or_else(|| GlslError::CreationError)?;
    context.shader_source(&shader, source);
    context.compile_shader(&shader);

    if context
        .get_shader_parameter(&shader, WebGl2RenderingContext::COMPILE_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(shader)
    } else {
        Err(context
            .get_shader_info_log(&shader)
            .map(GlslError::CompileError)
            .unwrap_or_else(|| {
                GlslError::CompileError("Unknown error creating shader".to_string())
            }))
    }
}
