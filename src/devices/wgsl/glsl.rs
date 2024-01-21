use naga::{
    back::glsl::{Options, PipelineOptions},
    proc::BoundsCheckPolicies,
    valid::ModuleInfo,
};

use super::error::TranslateError;

pub struct Glsl {}

impl Glsl {
    pub fn from_wgsl(src: impl AsRef<str>) {}
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
    use super::write_glsl;

    #[test]
    fn test_wgsl_to_glsl_translation() {
        let wgsl = "
            @fragment
            fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 0.0, 0.0, 1.0);
            }
        ";
    }
}
