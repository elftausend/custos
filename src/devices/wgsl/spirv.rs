use core::{mem::size_of_val, str::FromStr};

use naga::{
    back::spv::{Options, PipelineOptions},
    valid::ModuleInfo,
};

use super::{error::TranslateError, parse_and_output};

pub struct Spirv {
    words_of_entries: Vec<Vec<u32>>,
}

impl Spirv {
    #[inline]
    pub fn from_wgsl(src: impl AsRef<str>) -> Result<Self, TranslateError> {
        Ok(Spirv {
            words_of_entries: parse_and_output(src, write_spirv)?.1,
        })
    }

    #[inline]
    pub fn as_slice(&self, idx: usize) -> &[u32] {
        &self.words_of_entries[idx]
    }

    #[inline]
    pub fn as_byte_slice(&self, idx: usize) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.words_of_entries[idx].as_ptr() as *const u8,
                size_of_val(self.words_of_entries[idx].as_slice()),
            )
        }
    }
}

pub fn write_spirv(
    module: &naga::Module,
    info: &ModuleInfo,
    shader_stage: naga::ShaderStage,
    entry_point: &str,
) -> Result<Vec<u32>, TranslateError> {
    let mut words = Vec::new();

    let mut writer =
        naga::back::spv::Writer::new(&Options::default()).map_err(TranslateError::BackendSpv)?;
    writer
        .write(
            module,
            info,
            Some(&PipelineOptions {
                shader_stage,
                entry_point: entry_point.into(),
            }),
            &None,
            &mut words,
        )
        .map_err(TranslateError::BackendSpv)?;

    Ok(words)
}

impl FromStr for Spirv {
    type Err = TranslateError;

    #[inline]
    fn from_str(src: &str) -> Result<Self, Self::Err> {
        Self::from_wgsl(src)
    }
}
