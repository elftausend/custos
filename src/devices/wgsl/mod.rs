#[cfg(feature = "spv")]
mod spirv;
#[cfg(feature = "spv")]
pub use spirv::*;

#[cfg(feature = "glsl")]
mod glsl;
#[cfg(feature = "glsl")]
pub use glsl::*;

mod error;
mod wgsl_device;

use self::error::TranslateError;
use naga::{valid::ModuleInfo, Module, ShaderStage};

pub fn parse_and_output<O>(
    src: impl AsRef<str>,
    output_fn: fn(&Module, &ModuleInfo, ShaderStage, &str) -> Result<O, TranslateError>,
) -> Result<Vec<O>, TranslateError> {
    let (module, info) = parse_and_validate_wgsl(src.as_ref())?;

    module
        .entry_points
        .iter()
        .map(|entry_point| output_fn(&module, &info, entry_point.stage, &entry_point.name))
        .collect::<Result<Vec<_>, _>>()
}

pub fn parse_and_validate_wgsl(src: &str) -> Result<(naga::Module, ModuleInfo), TranslateError> {
    let mut frontend = naga::front::wgsl::Frontend::new();

    let module = frontend.parse(src).map_err(TranslateError::Frontend)?;

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );

    let info = validator
        .validate(&module)
        .map_err(TranslateError::Validate)?;
    Ok((module, info))
}
