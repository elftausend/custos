use naga::{
    back::glsl::{Options, PipelineOptions, Version},
    proc::BoundsCheckPolicies,
    valid::ModuleInfo,
    ShaderStage,
};
use web_sys::{WebGl2RenderingContext, WebGlShader};

use super::{error::TranslateError, parse_and_output};

mod shader;
pub use shader::*;

mod error;
pub use error::*;

#[derive(Debug, Clone)]
pub struct Glsl {
    pub sources: (naga::Module, Vec<ShaderSource>),
}

impl Glsl {
    #[inline]
    pub fn from_wgsl(src: impl AsRef<str>) -> Result<Self, TranslateError> {
        Ok(Glsl {
            sources: parse_and_output(src, write_glsl)?,
        })
    }

    #[inline]
    pub fn from_wgsl_compute(src: impl AsRef<str>) -> Result<Self, TranslateError> {
        Ok(Glsl {
            sources: parse_and_output(src, write_webgl_compute)?,
        })
    }

    #[inline]
    pub fn compile_all(
        &self,
        context: &WebGl2RenderingContext,
    ) -> Result<Vec<WebGlShader>, GlslError> {
        self.sources.1.iter().map(|s| s.compile(context)).collect()
    }
}

pub fn write_webgl_compute(
    module: &naga::Module,
    info: &ModuleInfo,
    shader_stage: naga::ShaderStage,
    entry_point: &str,
) -> Result<ShaderSource, TranslateError> {
    if shader_stage != ShaderStage::Compute {
        return Err(TranslateError::BackendGlsl(
            naga::back::glsl::Error::Custom("Shader must be a compute shader".into()),
        ));
    }

    let mut glsl = String::new();
    let options = Options {
        version: Version::Embedded {
            version: 310, // 310 is required for compute shaders (bypasses a check inside of naga)
            is_webgl: true,
        },
        ..Default::default()
    };
    let pipeline_options = PipelineOptions {
        shader_stage: ShaderStage::Compute,
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
    let reflection_info = Some(
        writer
            .write_webgl_compute()
            .map_err(TranslateError::BackendGlsl)?,
    );
    Ok(ShaderSource {
        shader_stage: ShaderStage::Fragment,
        src: glsl.replace("#version 310 es", "#version 300 es"),
        reflection_info,
    })
}

pub fn write_glsl(
    module: &naga::Module,
    info: &ModuleInfo,
    shader_stage: naga::ShaderStage,
    entry_point: &str,
) -> Result<ShaderSource, TranslateError> {
    let mut glsl = String::new();
    let options = Options {
        version: Version::new_gles(300),
        ..Default::default()
    };
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
    Ok(ShaderSource {
        shader_stage,
        src: glsl,
        reflection_info: None,
    })
}

#[cfg(test)]
mod tests {
    use naga::ShaderStage;

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
        // let is = format!("{glsl:?}");

        assert_eq!(glsl.sources.1[0].shader_stage, ShaderStage::Fragment);
        // assert_eq!(glsl.sources[0].src, r#"#version 300 es\n\nprecision highp float;\nprecision highp int;\n\nlayout(location = 0) out vec4 _fs2p_location0;\n\nvoid main() {\n    _fs2p_location0 = vec4(1.0, 0.0, 0.0, 1.0);\n    return;\n}\n\n"#);

        // let should = r##"Glsl { sources: [(shader_stage: Fragment, "#version 300 es\n\nprecision highp float;\nprecision highp int;\n\nlayout(location = 0) out vec4 _fs2p_location0;\n\nvoid main() {\n    _fs2p_location0 = vec4(1.0, 0.0, 0.0, 1.0);\n    return;\n}\n\n")] }"##;
        // assert_eq!(is, should);
    }

    #[test]
    fn test_wgsl_to_glsl_translation2() {
        let wgsl = "
            @vertex
            fn vs_main(@location(0) position: vec4<f32>) -> @builtin(position) vec4<f32> {
                return position;
            }

            // @vertex
            // fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4<f32> {
            //     let x = f32(i32(in_vertex_index) - 1);
            //     let y = f32(i32(in_vertex_index & 1u) * 2 - 1);
            //     return vec4<f32>(x, y, 0.0, 1.0);
            // }

            @fragment
            fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 0.0, 0.0, 1.0);
            }
        ";
        let glsl = Glsl::from_wgsl(wgsl).unwrap();
        println!("{}", glsl.sources.1[0].src);
        println!("{}", glsl.sources.1[1].src);
    }
}
