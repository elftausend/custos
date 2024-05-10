use std::rc::Rc;
use web_sys::{WebGl2RenderingContext, WebGlFramebuffer, WebGlTexture};

use super::context::Context;
use crate::{flag::AllocFlag, HasId, PtrType};

fn compute_texture_dimensions(length: usize) -> (usize, usize) {
    let sqrt = (length as f64).sqrt().ceil();
    (sqrt as usize, sqrt as usize)
}

#[derive(Debug)]
pub struct WebGlData {
    texture: WebGlTexture,
    texture_width: usize,
    texture_height: usize,
    len: usize,
    pub out_idx: Option<usize>,
    context: Rc<Context>,
    flag: AllocFlag,
    id: usize,
}

impl WebGlData {
    pub fn new(context: Rc<Context>, len: usize, flag: AllocFlag) -> Option<Self> {
        let texture = context.create_texture()?;
        let (texture_width, texture_height) = compute_texture_dimensions(len);

        context.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&texture));
        context.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_MIN_FILTER,
            WebGl2RenderingContext::NEAREST as i32,
        );
        context.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_MAG_FILTER,
            WebGl2RenderingContext::NEAREST as i32,
        );
        context.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_WRAP_S,
            WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
        );
        context.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_WRAP_T,
            WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
        );
        context.bind_texture(WebGl2RenderingContext::TEXTURE_2D, None);

        let mut buffer = WebGlData {
            texture,
            len,
            texture_width,
            texture_height,
            out_idx: None,
            id: context.gen_id(),
            context,
            flag,
        };
        buffer.write(&vec![0.; len])?;
        Some(buffer)
    }

    pub fn write(&mut self, data: &[f32]) -> Option<()> {
        let context = &self.context;

        context.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&self.texture));

        // let texture_data = vec![0u8; self.len * 4];
        assert_eq!(data.len(), self.len);
        let texture_data =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        unsafe {
            let texture_data = js_sys::Uint8Array::view(&texture_data);

            context.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_array_buffer_view(
                WebGl2RenderingContext::TEXTURE_2D,
                0,
                WebGl2RenderingContext::RGBA as i32,
                self.texture_width as i32,
                self.texture_height as i32,
                0,
                WebGl2RenderingContext::RGBA,
                WebGl2RenderingContext::UNSIGNED_BYTE,
                Some(&texture_data)
            ).ok()?;
        }
        context.bind_texture(WebGl2RenderingContext::TEXTURE_2D, None);
        Some(())
    }

    pub fn read(&self, frame_buf: &WebGlFramebuffer, color_attachment_idx: u32) -> Vec<f32> {
        let context = &self.context;

        context.bind_framebuffer(WebGl2RenderingContext::FRAMEBUFFER, Some(&frame_buf));
        context.framebuffer_texture_2d(
            WebGl2RenderingContext::FRAMEBUFFER,
            WebGl2RenderingContext::COLOR_ATTACHMENT0 + color_attachment_idx,
            WebGl2RenderingContext::TEXTURE_2D,
            Some(&self.texture),
            0,
        );

        assert_eq!(
            context.check_framebuffer_status(WebGl2RenderingContext::FRAMEBUFFER),
            WebGl2RenderingContext::FRAMEBUFFER_COMPLETE
        );

        let mut read_data = vec![0.; self.len];
        let texture_data = unsafe {
            std::slice::from_raw_parts_mut(read_data.as_mut_ptr() as *mut u8, read_data.len() * 4)
        };

        context
            .read_pixels_with_u8_array_and_dst_offset(
                0,
                0,
                self.texture_width as i32,
                self.texture_height as i32,
                WebGl2RenderingContext::RGBA,
                WebGl2RenderingContext::UNSIGNED_BYTE,
                texture_data,
                0,
            )
            .unwrap();

        context.bind_framebuffer(WebGl2RenderingContext::FRAMEBUFFER, None);
        read_data
    }
}

impl Drop for WebGlData {
    #[inline]
    fn drop(&mut self) {
        if !self.flag.continue_deallocation() {
            return;
        }
        self.context.delete_texture(Some(&self.texture));
    }
}

impl PtrType for WebGlData {
    #[inline]
    fn size(&self) -> usize {
        self.len
    }

    #[inline]
    fn flag(&self) -> crate::flag::AllocFlag {
        self.flag
    }

    #[inline]
    unsafe fn set_flag(&mut self, flag: crate::flag::AllocFlag) {
        self.flag = flag;
    }
}

impl HasId for WebGlData {
    #[inline]
    fn id(&self) -> crate::Id {
        crate::Id {
            id: self.id as u64,
            len: self.len,
        }
    }
}
