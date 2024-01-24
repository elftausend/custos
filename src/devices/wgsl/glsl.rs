use naga::{
    back::glsl::{Options, PipelineOptions},
    proc::BoundsCheckPolicies,
    valid::ModuleInfo, ShaderStage,
};
use web_sys::{WebGl2RenderingContext, WebGlShader};

use super::{parse_and_output, error::TranslateError};

#[derive(Debug, Clone)]
pub struct Glsl {
    pub sources: Vec<Shader>
}

impl Glsl {
    #[inline]
    pub fn from_wgsl(src: impl AsRef<str>) -> Result<Self, TranslateError> {
        Ok(Glsl {
            sources: parse_and_output(src, write_glsl)?,
        })
    }

    #[inline]
    pub fn compile_all(&self, context: &WebGl2RenderingContext) -> Result<Vec<WebGlShader>, String> {
        self.sources.iter().map(|s| s.compile(context)).collect()
    }
}

pub fn write_glsl(
    module: &naga::Module,
    info: &ModuleInfo,
    shader_stage: naga::ShaderStage,
    entry_point: &str,
) -> Result<Shader, TranslateError> {
    let mut glsl = String::new();
    let options = Options::default();
    let pipeline_options = PipelineOptions {
        shader_stage,
        entry_point: entry_point.into(),
        multiview: None,
    };

    let mut writer = naga::back::glsl::Writer::new(
        &mut glsl,
        module,
        info,
        &options,
        &pipeline_options,
        BoundsCheckPolicies::default(),
    )
    .map_err(TranslateError::BackendGlsl)?;
    writer.write().map_err(TranslateError::BackendGlsl)?;
    Ok(Shader {
        shader_stage,
        src: glsl
    })
}

#[derive(Debug, Clone)]
pub struct Shader {
    pub shader_stage: ShaderStage,
    pub src: String
}

impl Shader {
    pub fn compile(&self, context: &WebGl2RenderingContext) -> Result<WebGlShader, String> {
        let shader_type = match self.shader_stage {
            ShaderStage::Vertex => WebGl2RenderingContext::VERTEX_SHADER,
            ShaderStage::Fragment => WebGl2RenderingContext::FRAGMENT_SHADER,
            _ => panic!("Unsupported shader stage: {:?}", self.shader_stage)
        };
        compile_shader(context, shader_type, &self.src)
    }
}


pub fn compile_shader(
    context: &WebGl2RenderingContext,
    shader_type: u32,
    source: &str,
) -> Result<WebGlShader, String> {
    let shader = context
        .create_shader(shader_type)
        .ok_or_else(|| String::from("Unable to create shader object"))?;
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
            .unwrap_or_else(|| String::from("Unknown error creating shader")))
    }
}

#[cfg(test)]
mod tests {
    use super::Glsl;

    #[test]
    fn test_wgsl_to_glsl_translation() {
        let wgsl = "
            @fragment
            fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 0.0, 0.0, 1.0);
            }
        ";
        let glsl = Glsl::from_wgsl(wgsl).unwrap();
        let is = format!("{glsl:?}");
        let should = r##"Glsl { sources: [(Fragment, "#version 310 es\n\nprecision highp float;\nprecision highp int;\n\nlayout(location = 0) out vec4 _fs2p_location0;\n\nvoid main() {\n    _fs2p_location0 = vec4(1.0, 0.0, 0.0, 1.0);\n    return;\n}\n\n")] }"##;
        assert_eq!(is, should);
    }
    
    #[test]
    fn test_wgsl_to_glsl_translation2() {
        let wgsl = "
            @vertex
            fn vs_main(@location(0) position: vec4<f32>) -> @builtin(position) vec4<f32> {
                return position;
            }
            @fragment
            fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 0.0, 0.0, 1.0);
            }
        ";
        let glsl = Glsl::from_wgsl(wgsl).unwrap();
        println!("{}", glsl.sources[0].src);
    }
}
