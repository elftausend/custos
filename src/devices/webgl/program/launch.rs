use js_sys::{wasm_bindgen::JsValue, Uint32Array};
use web_sys::{WebGl2RenderingContext, WebGlBuffer, WebGlFramebuffer};

use crate::webgl::data::WebGlData;

use super::Program;

impl Program {
    pub fn launch(
        &self,
        frame_buf: &WebGlFramebuffer,
        indices_buffer: &WebGlBuffer,
        args: &[&WebGlData<f32>],
    ) -> crate::Result<()> {
        let program = &self.program;
        let reflection_info = &self.reflection_info;

        let output_storage_layout_names = reflection_info.outputs.iter().collect::<Vec<_>>();

        let input_storage_uniform_names = reflection_info
            .input_storage_uniforms
            .values()
            .collect::<Vec<_>>();

        let out_binding_idxs = output_storage_layout_names
            .iter()
            .map(|(handle, _)| {
                self.module.global_variables[**handle]
                    .binding
                    .as_ref()
                    .unwrap()
                    .binding
            })
            .collect::<Vec<_>>();

        let mut out_idxs = vec![];

        for idx in 0..args.len() {
            if out_binding_idxs.contains(&(idx as u32)) {
                out_idxs.push(idx);
            }
        }

        let first_arg = &args[out_idxs[0]];
        let (first_th, first_tw) = (first_arg.texture_height, first_arg.texture_width);

        // TODO: convert to error
        assert!(out_idxs
            .iter()
            .map(|idx| {
                let arg = &args[*idx];
                (arg.texture_height, arg.texture_width)
            })
            .all(|(th, tw)| th == first_th && tw == first_tw));

        let out_bufs = out_idxs.iter().map(|idx| &args[*idx]).collect::<Vec<_>>();
        let input_bufs = args
            .iter()
            .enumerate()
            .filter(|(idx, _arg)| !out_idxs.contains(idx))
            .map(|(_, data)| data)
            .collect::<Vec<_>>();

        let context = &self.context;
        let thread_viewport_width_uniform = context
            .get_uniform_location(program, "thread_viewport_width")
            .ok_or("cannot find thread vpw")?;
        let thread_viewport_height_uniform = context
            .get_uniform_location(program, "thread_viewport_height")
            .ok_or("cannot find thread vpw")?;

        // do not bubble up error -> it is possible that the internal glsl compiler removes unused uniforms
        let gws_x_uniform = context.get_uniform_location(program, "gws_x");
        let gws_y_uniform = context.get_uniform_location(program, "gws_y");
        let gws_z_uniform = context.get_uniform_location(program, "gws_z");

        let mut input_uniforms = Vec::with_capacity(input_storage_uniform_names.len());

        // TODO: support e.g. floats as inputs
        for uniform_name in input_storage_uniform_names {
            input_uniforms.push([
                context
                    .get_uniform_location(program, uniform_name)
                    .ok_or("cannot find uniform input")?,
                context
                    .get_uniform_location(program, &format!("{uniform_name}_texture_width"))
                    .ok_or("cannot find uniform input width")?,
                context
                    .get_uniform_location(program, &format!("{uniform_name}_texture_height"))
                    .ok_or("cannot find uniform input height")?,
            ]);
        }

        let mut output_size_uniforms = Vec::with_capacity(output_storage_layout_names.len());

        for (_, uniform_name) in &output_storage_layout_names {
            output_size_uniforms.push([
                context
                    .get_uniform_location(program, &format!("{uniform_name}_texture_width"))
                    .ok_or("cannot find uniform out width")?,
                context
                    .get_uniform_location(program, &format!("{uniform_name}_texture_height"))
                    .ok_or("cannot find uniform out height")?,
            ]);
        }

        let color_attachments = Uint32Array::new(&JsValue::from(output_storage_layout_names.len()));

        for idx in 0..output_storage_layout_names.len() as u32 {
            let attachment = WebGl2RenderingContext::COLOR_ATTACHMENT0 + idx;
            color_attachments.set_index(idx, attachment);
        }

        context.bind_framebuffer(WebGl2RenderingContext::FRAMEBUFFER, Some(frame_buf));
        context.draw_buffers(&color_attachments);

        let mut color_idx = 0;
        for idx in &out_idxs {
            let arg = &args[*idx];
            context.framebuffer_texture_2d(
                WebGl2RenderingContext::FRAMEBUFFER,
                WebGl2RenderingContext::COLOR_ATTACHMENT0 + color_idx,
                WebGl2RenderingContext::TEXTURE_2D,
                Some(&arg.texture),
                0,
            );
            color_idx += 1;
        }

        assert_eq!(
            context.check_framebuffer_status(WebGl2RenderingContext::FRAMEBUFFER),
            WebGl2RenderingContext::FRAMEBUFFER_COMPLETE
        );

        context.use_program(Some(&program));

        context.viewport(0, 0, first_tw as i32, first_th as i32);
        context.uniform1ui(Some(&thread_viewport_width_uniform), first_tw as u32);
        context.uniform1ui(Some(&thread_viewport_height_uniform), first_th as u32);

        context.uniform1ui(gws_x_uniform.as_ref(), first_tw as u32 * first_th as u32);
        context.uniform1ui(gws_y_uniform.as_ref(), 1);
        context.uniform1ui(gws_z_uniform.as_ref(), 1);

        for (idx, (input_uniform, gl_buf)) in input_uniforms.iter().zip(input_bufs).enumerate() {
            context.uniform1i(Some(&input_uniform[0]), idx as i32);
            context.uniform1ui(Some(&input_uniform[1]), gl_buf.texture_width as u32);
            context.uniform1ui(Some(&input_uniform[2]), gl_buf.texture_height as u32);
            context.active_texture(WebGl2RenderingContext::TEXTURE0 + idx as u32);
            context.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&gl_buf.texture))
        }

        for (_idx, (output_size_uniform, gl_buf)) in
            output_size_uniforms.iter().zip(out_bufs).enumerate()
        {
            context.uniform1ui(Some(&output_size_uniform[0]), gl_buf.texture_width as u32);
            context.uniform1ui(Some(&output_size_uniform[1]), gl_buf.texture_height as u32);
        }

        context.bind_buffer(
            WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
            Some(indices_buffer),
        );

        context.draw_elements_with_i32(
            WebGl2RenderingContext::TRIANGLES,
            6,
            WebGl2RenderingContext::UNSIGNED_SHORT,
            0,
        );

        for idx in 0..color_idx {
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
