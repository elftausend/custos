use core::marker::PhantomData;
use std::rc::Rc;
use web_sys::{WebGl2RenderingContext, WebGlFramebuffer, WebGlTexture};

use super::{context::Context, program::WebGlNumber};
use crate::{flag::AllocFlag, HasId, PtrType};

fn compute_texture_dimensions(length: usize) -> (usize, usize) {
    let sqrt = (length as f64).sqrt().ceil();
    (sqrt as usize, sqrt as usize)
}

#[derive(Debug)]
pub struct WebGlData<T> {
    pub texture: WebGlTexture,
    pub texture_width: usize,
    pub texture_height: usize,
    pub len: usize,
    pub out_idx: Option<u32>,
    context: Rc<Context>,
    flag: AllocFlag,
    id: usize,
    _pd: PhantomData<T>,
}

impl<T> crate::ShallowCopy for WebGlData<T> {
    unsafe fn shallow(&self) -> Self {
        Self {
            texture: self.texture.clone(),
            texture_width: self.texture_width,
            texture_height: self.texture_height,
            len: self.len,
            out_idx: self.out_idx,
            context: self.context.clone(),
            flag: AllocFlag::Wrapper,
            id: self.id,
            _pd: PhantomData,
        }
    }
}

impl<T: WebGlNumber> WebGlData<T> {
    pub fn new(context: Rc<Context>, len: usize, flag: AllocFlag) -> Option<Self>
    where
        T: Default,
    {
        let texture = context.create_texture()?;
        let (texture_width, texture_height) = compute_texture_dimensions(len);
        // let (texture_width, texture_height) = (len, 1);

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
            _pd: PhantomData,
        };
        buffer.write((0..len).map(|_| T::default()))?;
        Some(buffer)
    }

    pub fn write(&mut self, data: impl IntoIterator<Item = T>) -> Option<()>
    where
        T: Default,
    {
        let context = &self.context;

        context.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&self.texture));

        // let texture_data = vec![0u8; self.len * 4];
        // assert_eq!(data.len(), self.len);

        let mut upload_data = Vec::with_capacity(self.texture_width * self.texture_height);
        upload_data.extend(data);

        assert_eq!(upload_data.len(), self.len);

        // padding
        upload_data
            .extend((0..self.texture_width * self.texture_height - self.len).map(|_| T::default()));

        unsafe {
            let texture_data = T::array_view(&upload_data);
            context.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_array_buffer_view(
                WebGl2RenderingContext::TEXTURE_2D,
                0,
                T::INTERNAL_FORMAT as i32,
                self.texture_width as i32,
                self.texture_height as i32,
                0,
                T::FORMAT,
                T::TYPE,
                Some(&texture_data)
            ).ok()?;
        }
        context.bind_texture(WebGl2RenderingContext::TEXTURE_2D, None);
        Some(())
    }

    pub fn read(&self, frame_buf: &WebGlFramebuffer, color_attachment_idx: u32) -> Vec<T>
    where
        T: Default + Clone,
    {
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

        let mut read_data = T::read_pixels(&context, self.texture_width, self.texture_height);
        context.bind_framebuffer(WebGl2RenderingContext::FRAMEBUFFER, None);
        read_data.truncate(self.len);
        read_data
    }
}

impl<T> Drop for WebGlData<T> {
    #[inline]
    fn drop(&mut self) {
        if !self.flag.continue_deallocation() {
            return;
        }
        self.context.delete_texture(Some(&self.texture));
    }
}

impl<T> PtrType for WebGlData<T> {
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

impl<T> HasId for WebGlData<T> {
    #[inline]
    fn id(&self) -> crate::Id {
        crate::Id {
            id: self.id as u64,
            len: self.len,
        }
    }
}