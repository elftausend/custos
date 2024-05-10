use std::rc::Rc;
use web_sys::{WebGl2RenderingContext, WebGlBuffer, WebGlProgram};

use crate::webgl::error::WebGlError;

use self::context::Context;

mod context;
mod data;
mod error;
mod webgl_device;

pub use webgl_device::*;

pub struct VertexAttributes {
    position: WebGlBuffer,
    texcoords: WebGlBuffer,
    context: Rc<Context>,
}

impl VertexAttributes {
    pub fn new(context: Rc<Context>, program: &WebGlProgram) -> crate::Result<Self> {
        #[rustfmt::skip]
        let vertices: [f32; 12] = [
            -1.0,-1.0, 0.0,
            -1.0, 1.0, 0.0,
            1.0, 1.0, 0.0,
            1.0,-1.0, 0.0
        ];

        let position_attribute_location = context.get_attrib_location(program, "position");
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

        // let vao = context
        //     .create_vertex_array()
        //     .ok_or("Could not create vertex array object")?;
        // context.bind_vertex_array(Some(&vao));

        context.enable_vertex_attrib_array(position_attribute_location as u32);
        context.vertex_attrib_pointer_with_i32(
            position_attribute_location as u32,
            3,
            WebGl2RenderingContext::FLOAT,
            false,
            0,
            0,
        );

        // context.bind_vertex_array(Some(&vao));

        #[rustfmt::skip]
        let tex_coords: [f32; 8] = [
            0.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
            1.0, 0.0
        ];

        let texcoords_buffer = context.create_buffer().ok_or(WebGlError::BufferCreation)?;
        let texcoords_attribute_location = context.get_attrib_location(program, "texcoords");
        context.bind_buffer(
            WebGl2RenderingContext::ARRAY_BUFFER,
            Some(&texcoords_buffer),
        );
        context.enable_vertex_attrib_array(texcoords_attribute_location as u32);
        context.vertex_attrib_pointer_with_i32(
            texcoords_attribute_location as u32,
            2,
            WebGl2RenderingContext::FLOAT,
            false,
            0,
            0,
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

        let indices_buffer = context
            .create_buffer()
            .ok_or(WebGlError::BufferCreation)?;
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
        todo!()
    }
}

impl Drop for VertexAttributes {
    #[inline]
    fn drop(&mut self) {
        self.context.delete_buffer(Some(&self.position));
        self.context.delete_buffer(Some(&self.texcoords));
    }
}
