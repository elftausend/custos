use std::rc::Rc;
use web_sys::{WebGl2RenderingContext, WebGlBuffer, WebGlProgram};

use crate::webgl::error::WebGlError;

use super::context::Context;

pub struct VertexAttributes {
    position_buffer: WebGlBuffer,
    texcoords_buffer: WebGlBuffer,
    indices_buffer: WebGlBuffer,
    context: Rc<Context>,
}

impl VertexAttributes {
    pub fn new(context: Rc<Context>) -> crate::Result<Self> {
        #[rustfmt::skip]
        let vertices: [f32; 12] = [
            -1.0,-1.0, 0.0,
            -1.0, 1.0, 0.0,
            1.0, 1.0, 0.0,
            1.0,-1.0, 0.0
        ];

        let position_buffer = context.create_buffer().ok_or(WebGlError::BufferCreation)?;
        context.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&position_buffer));

        // Note that `Float32Array::view` is somewhat dangerous (hence the
        // `unsafe`!). This is creating a raw view into our module's
        // `WebAssembly.Memory` buffer, but if we allocate more pages for ourself
        // (aka do a memory allocation in Rust) it'll cause the buffer to change,
        // causing the `Float32Array` to be invalid.
        //
        // As a result, after `Float32Array::view` we have to be very careful not to
        // do any memory allocations before it's dropped.
        unsafe {
            let positions_array_buf_view = js_sys::Float32Array::view(&vertices);

            context.buffer_data_with_array_buffer_view(
                WebGl2RenderingContext::ARRAY_BUFFER,
                &positions_array_buf_view,
                WebGl2RenderingContext::STATIC_DRAW,
            );
        }

        #[rustfmt::skip]
        let tex_coords: [f32; 8] = [
            0.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
            1.0, 0.0
        ];

        let texcoords_buffer = context.create_buffer().ok_or(WebGlError::BufferCreation)?;
        context.bind_buffer(
            WebGl2RenderingContext::ARRAY_BUFFER,
            Some(&texcoords_buffer),
        );

        unsafe {
            let positions_array_buf_view = js_sys::Float32Array::view(&tex_coords);

            context.buffer_data_with_array_buffer_view(
                WebGl2RenderingContext::ARRAY_BUFFER,
                &positions_array_buf_view,
                WebGl2RenderingContext::STATIC_DRAW,
            );
        }

        let indices: [u16; 6] = [0, 1, 2, 2, 3, 0];

        let indices_buffer = context.create_buffer().ok_or(WebGlError::BufferCreation)?;
        context.bind_buffer(
            WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
            Some(&indices_buffer),
        );

        unsafe {
            let positions_array_buf_view = js_sys::Uint16Array::view(&indices);

            context.buffer_data_with_array_buffer_view(
                WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
                &positions_array_buf_view,
                WebGl2RenderingContext::STATIC_DRAW,
            );
        }

        Ok(VertexAttributes {
            position_buffer,
            texcoords_buffer,
            indices_buffer,
            context,
        })
    }

    pub fn bind(&self, program: &WebGlProgram) -> crate::Result<()> {
        let context = &self.context;

        context.bind_buffer(
            WebGl2RenderingContext::ARRAY_BUFFER,
            Some(&self.position_buffer),
        );
        let position_attribute_location = context.get_attrib_location(program, "position");

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
            Some(&self.texcoords_buffer),
        );
        let texcoords_attribute_location = context.get_attrib_location(program, "texcoords");
        context.enable_vertex_attrib_array(texcoords_attribute_location as u32);
        context.vertex_attrib_pointer_with_i32(
            texcoords_attribute_location as u32,
            2,
            WebGl2RenderingContext::FLOAT,
            false,
            0,
            0,
        );

        Ok(())
    }

    #[inline]
    pub fn position_buffer(&self) -> &WebGlBuffer {
        &self.position_buffer
    }

    #[inline]
    pub fn texcoords_buffer(&self) -> &WebGlBuffer {
        &self.texcoords_buffer
    }

    #[inline]
    pub fn indices_buffer(&self) -> &WebGlBuffer {
        &self.indices_buffer
    }
}

impl Drop for VertexAttributes {
    #[inline]
    fn drop(&mut self) {
        self.context.delete_buffer(Some(&self.position_buffer));
        self.context.delete_buffer(Some(&self.texcoords_buffer));
    }
}
