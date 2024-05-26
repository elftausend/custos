use js_sys::{wasm_bindgen::JsValue, Uint32Array};
use web_sys::{WebGl2RenderingContext, WebGlFramebuffer};

use crate::webgl::{error::WebGlError, vertex_attributes::VertexAttributes};

use super::{shader_argument::AsWebGlShaderArgument, Program};

impl Program {
    ///
    pub fn launch(
        &self,
        frame_buf: &WebGlFramebuffer,
        vertex_attributes: &VertexAttributes,
        args: &[&dyn AsWebGlShaderArgument],
        gws: [u32; 3],
    ) -> crate::Result<()> {
        let program = &self.program;
        let reflection_info = &self.reflection_info;

        if args.len()
            != reflection_info.input_storage_uniforms.len()
                + reflection_info.other_uniforms.len()
                + reflection_info.outputs.len()
        {
            return Err(WebGlError::ArgumentCountMismatch.into());
        }

        let output_storage_layout_names = reflection_info.outputs.iter().collect::<Vec<_>>();

        let input_storage_uniform_names = reflection_info
            .input_storage_uniforms
            .values()
            .collect::<Vec<_>>();

        let out_idxs = output_storage_layout_names
            .iter()
            .map(|(handle, _)| {
                self.module.global_variables[**handle]
                    .binding
                    .as_ref()
                    .unwrap()
                    .binding as usize
            })
            .collect::<Vec<_>>();

        let input_idxs = reflection_info
            .input_storage_uniforms
            .iter()
            .map(|(handle, _)| {
                self.module.global_variables[*handle]
                    .binding
                    .as_ref()
                    .unwrap()
                    .binding as usize
            })
            .collect::<Vec<_>>();

        let first_arg = &args[out_idxs[0]];
        let (first_th, first_tw) = (first_arg.texture_height(), first_arg.texture_width());

        if !out_idxs
            .iter()
            .map(|idx| {
                let arg = &args[*idx];
                (arg.texture_height(), arg.texture_width())
            })
            .all(|(th, tw)| th == first_th && tw == first_tw)
        {
            return Err(WebGlError::OutputBufferSizeMismatch.into());
        }

        let out_bufs = out_idxs.iter().map(|idx| &args[*idx]).collect::<Vec<_>>();
        let input_bufs = input_idxs.iter().map(|idx| &args[*idx]).collect::<Vec<_>>();

        let other_inputs = args
            .iter()
            .enumerate()
            .filter(|(idx, _arg)| !out_idxs.contains(idx) && !input_idxs.contains(idx))
            .map(|(_, data)| data)
            .collect::<Vec<_>>();

        let context = &self.context;
        let thread_viewport_width_uniform = context
            .get_uniform_location(program, "thread_viewport_width");
        let thread_viewport_height_uniform = context
            .get_uniform_location(program, "thread_viewport_height");

        // do not bubble up error -> it is possible that the internal glsl compiler removes unused uniforms
        let gws_x_uniform = context.get_uniform_location(program, "gws_x");
        let gws_y_uniform = context.get_uniform_location(program, "gws_y");
        let gws_z_uniform = context.get_uniform_location(program, "gws_z");

        let mut input_uniforms = Vec::with_capacity(input_storage_uniform_names.len());

        // TODO: support e.g. floats as inputs
        for uniform_name in input_storage_uniform_names {
            input_uniforms.push([
                context.get_uniform_location(program, uniform_name),
                context.get_uniform_location(program, &format!("{uniform_name}_texture_width")),
                context.get_uniform_location(program, &format!("{uniform_name}_texture_height")),
            ]);
        }

        let mut output_size_uniforms = Vec::with_capacity(output_storage_layout_names.len());

        for (_, uniform_name) in &output_storage_layout_names {
            output_size_uniforms.push([
                context.get_uniform_location(program, &format!("{uniform_name}_texture_width")),
                context.get_uniform_location(program, &format!("{uniform_name}_texture_height")),
            ]);
        }

        let position_attribute_location = context.get_attrib_location(&program, "position");
        context.bind_buffer(
            WebGl2RenderingContext::ARRAY_BUFFER,
            Some(vertex_attributes.position_buffer()),
        );

        context.enable_vertex_attrib_array(position_attribute_location as u32);
        context.vertex_attrib_pointer_with_i32(
            position_attribute_location as u32,
            3,
            WebGl2RenderingContext::FLOAT,
            false,
            0,
            0,
        );

        context.bind_buffer(
            WebGl2RenderingContext::ARRAY_BUFFER,
            Some(vertex_attributes.texcoords_buffer()),
        );
        let texcoords_attribute_location = context.get_attrib_location(&program, "texcoords");
        context.enable_vertex_attrib_array(texcoords_attribute_location as u32);
        context.vertex_attrib_pointer_with_i32(
            texcoords_attribute_location as u32,
            2,
            WebGl2RenderingContext::FLOAT,
            false,
            0,
            0,
        );

        let color_attachments = Uint32Array::new(&JsValue::from(output_storage_layout_names.len()));

        for idx in 0..output_storage_layout_names.len() as u32 {
            let attachment = WebGl2RenderingContext::COLOR_ATTACHMENT0 + idx;
            color_attachments.set_index(idx, attachment);
        }

        context.bind_framebuffer(WebGl2RenderingContext::FRAMEBUFFER, Some(frame_buf));
        context.draw_buffers(&color_attachments);

        for (color_idx, out_buf) in out_bufs.iter().enumerate() {
            context.framebuffer_texture_2d(
                WebGl2RenderingContext::FRAMEBUFFER,
                WebGl2RenderingContext::COLOR_ATTACHMENT0 + color_idx as u32,
                WebGl2RenderingContext::TEXTURE_2D,
                Some(&out_buf.texture().unwrap()),
                0,
            );
        }

        assert_eq!(
            context.check_framebuffer_status(WebGl2RenderingContext::FRAMEBUFFER),
            WebGl2RenderingContext::FRAMEBUFFER_COMPLETE
        );

        context.use_program(Some(program));

        context.viewport(0, 0, first_tw as i32, first_th as i32);
        context.uniform1ui(thread_viewport_width_uniform.as_ref(), first_tw as u32);
        context.uniform1ui(thread_viewport_height_uniform.as_ref(), first_th as u32);

        context.uniform1ui(gws_x_uniform.as_ref(), gws[0]);
        context.uniform1ui(gws_y_uniform.as_ref(), gws[1]);
        context.uniform1ui(gws_z_uniform.as_ref(), gws[2]);

        for (idx, (input_uniform, gl_buf)) in input_uniforms.iter().zip(input_bufs).enumerate() {
            context.uniform1i(input_uniform[0].as_ref(), idx as i32);
            context.uniform1ui(input_uniform[1].as_ref(), gl_buf.texture_width() as u32);
            context.uniform1ui(input_uniform[2].as_ref(), gl_buf.texture_height() as u32);
            context.active_texture(WebGl2RenderingContext::TEXTURE0 + idx as u32);
            context.bind_texture(
                WebGl2RenderingContext::TEXTURE_2D,
                Some(&gl_buf.texture().unwrap()),
            )
        }

        for (output_size_uniform, gl_buf) in output_size_uniforms.iter().zip(out_bufs) {
            context.uniform1ui(
                output_size_uniform[0].as_ref(),
                gl_buf.texture_width() as u32,
            );
            context.uniform1ui(
                output_size_uniform[1].as_ref(),
                gl_buf.texture_height() as u32,
            );
        }

        // other uniforms
        for (value_arg, (handle, global_name)) in other_inputs
            .iter()
            .zip(reflection_info.other_uniforms.iter())
        {
            let global = &self.module.global_variables[*handle];
            let ty = &self.module.types[global.ty];
            let uniform_location = context.get_uniform_location(program, &global_name);
            value_arg.set_num_uniform(context, uniform_location.as_ref(), ty)?;
        }

        context.bind_buffer(
            WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
            Some(vertex_attributes.indices_buffer()),
        );

        context.draw_elements_with_i32(
            WebGl2RenderingContext::TRIANGLES,
            6,
            WebGl2RenderingContext::UNSIGNED_SHORT,
            0,
        );

        for idx in 0..out_idxs.len() as u32 {
            context.framebuffer_texture_2d(
                WebGl2RenderingContext::FRAMEBUFFER,
                WebGl2RenderingContext::COLOR_ATTACHMENT0 + idx,
                WebGl2RenderingContext::TEXTURE_2D,
                None,
                0,
            );
        }

        context.bind_framebuffer(WebGl2RenderingContext::FRAMEBUFFER, None);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{Base, Buffer, Device, WebGL};

    #[ignore]
    #[test]
    fn test_launch_shader_webgl() {
        let device = WebGL::<Base>::new().unwrap();
        let x = device.buffer([2.; 16]);
        let mut out: Buffer<f32, _> = device.buffer(x.len);
        // out.base_mut().write(&[5.; 16]);

        let src = "
            @group(0)
            @binding(0)
            var<storage, read> x: array<f32>;

            @group(0)
            @binding(1)
            var<storage, read_write> out: array<f32>;
            
            @compute
            @workgroup_size(32)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                if global_id.x >= arrayLength(&out) {
                    return;    
                }

                var counter = 0.0;
                for (var i = 0u; i < 10u; i++) {
                    counter += 1.0;
                }

                // if out is used on the right side: problem at the moment
                out[global_id.x] = counter * x[global_id.x];
                // out[global_id.x] = f32(global_id.x) + x[global_id.x] * 0.00000001;
                // out[global_id.x] = 3.0;
            }
        ";

        let err = device.launch_shader(src, [x.len() as u32, 1, 1], &[&x, &out, &7]);
    }
}
