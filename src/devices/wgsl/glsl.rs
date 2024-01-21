use naga::{
    back::glsl::{Options, PipelineOptions},
    proc::BoundsCheckPolicies,
    valid::ModuleInfo,
};

use super::{parse_and_output, error::TranslateError};

#[derive(Debug, Clone)]
pub struct Glsl {
    pub sources: Vec<String>
}

impl Glsl {
    pub fn from_wgsl(src: impl AsRef<str>) -> Result<Self, TranslateError> {
        Ok(Glsl {
            sources: parse_and_output(src, write_glsl)?,
        })
    }
}

pub fn write_glsl(
    module: &naga::Module,
    info: &ModuleInfo,
    shader_stage: naga::ShaderStage,
    entry_point: &str,
) -> Result<String, TranslateError> {
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
    Ok(glsl)
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
        let should = r##"Glsl { sources: ["#version 310 es\n\nprecision highp float;\nprecision highp int;\n\nlayout(location = 0) out vec4 _fs2p_location0;\n\nvoid main() {\n    _fs2p_location0 = vec4(1.0, 0.0, 0.0, 1.0);\n    return;\n}\n\n"] }"##;
        assert_eq!(is, should);
    }
}
