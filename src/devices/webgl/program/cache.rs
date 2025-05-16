use super::Program;
use crate::webgl::context::Context;
use std::collections::HashMap;
use std::rc::Rc;
use web_sys::WebGlShader;

#[derive(Debug, Default)]
pub struct ProgramCache {
    cache: HashMap<String, Program>,
}

impl ProgramCache {
    #[inline]
    pub fn new() -> Self {
        Self {
            cache: Default::default(),
        }
    }

    pub fn add(
        &mut self,
        context: Rc<Context>,
        vert_shader: &WebGlShader,
        src: impl AsRef<str>,
    ) -> crate::Result<&Program> {
        let program = Program::new(context, &vert_shader, &src)?;
        self.cache.insert(src.as_ref().to_string(), program);
        Ok(self.cache.get(src.as_ref()).unwrap())
    }

    #[inline]
    pub fn get<'a>(
        &'a mut self,
        context: Rc<Context>,
        vert_shader: &WebGlShader,
        src: impl AsRef<str>,
    ) -> crate::Result<&'a Program> {
        if !self.cache.contains_key(src.as_ref()) {
            return self.add(context, vert_shader, &src);
        }
        Ok(self.cache.get(src.as_ref()).unwrap())
    }
}
