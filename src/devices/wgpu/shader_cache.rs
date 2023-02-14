use std::borrow::Cow;
use std::collections::HashMap;
use wgpu::ShaderModule;

#[derive(Debug, Default)]
pub struct ShaderCache {
    shaders: HashMap<String, ShaderModule>,
}

impl ShaderCache {
    pub fn shader(&mut self, device: &wgpu::Device, src: &str) -> &ShaderModule {
        self.add_shader(device, src);
        self.shaders.get(src).unwrap()
    }

    fn add_shader(&mut self, device: &wgpu::Device, src: &str) {
        if self.shaders.get(src).is_some() {
            return;
        }

        let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(src)),
        });

        self.shaders.insert(src.to_string(), cs_module);
    }
}
