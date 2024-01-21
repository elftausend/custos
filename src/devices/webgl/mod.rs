use js_sys::wasm_bindgen::JsValue;
use web_sys::Element;

use crate::{
    impl_device_traits, Alloc, Base, Buffer, Device, Module, Num, OnDropBuffer, Setup, Shape,
    WrappedData,
};

use self::context::Context;

mod context;

pub struct WebGL<Mods = Base> {
    pub modules: Mods,
    pub context: Context,
}
impl<SimpleMods> WebGL<SimpleMods> {
    #[inline]
    pub fn new<NewMods>(maybe_canvas: Element) -> Result<WebGL<SimpleMods::Module>, JsValue>
    where
        SimpleMods: Module<WebGL, Module = NewMods>,
        NewMods: Setup<WebGL<NewMods>>,
    {
        let mut webgl = WebGL {
            modules: SimpleMods::new(),
            context: Context::new(maybe_canvas)?,
        };
        NewMods::setup(&mut webgl).unwrap();
        Ok(webgl)
    }
}

impl_device_traits!(WebGL);

// impl<Mods> OnDropBuffer for WebGL<Mods> {
//     fn on_drop_buffer<T, D: Device, S: Shape>(&self, _device: &D, _buf: &Buffer<T, D, S>) {}
// }

impl<Mods: OnDropBuffer> Device for WebGL<Mods> {
    type Base<T, S: crate::Shape> = Num<T>;
    type Data<T, S: Shape> = Self::Wrap<T, Self::Base<T, S>>;

    type Error = JsValue;

    #[inline(always)]
    fn base_to_data<T, S: crate::Shape>(&self, base: Self::Base<T, S>) -> Self::Data<T, S> {
        self.wrap_in_base(base)
    }

    #[inline(always)]
    fn wrap_to_data<T, S: crate::Shape>(
        &self,
        wrap: Self::Wrap<T, Self::Base<T, S>>,
    ) -> Self::Data<T, S> {
        wrap
    }

    #[inline(always)]
    fn data_as_wrap<'a, T, S: crate::Shape>(
        data: &'a Self::Data<T, S>,
    ) -> &'a Self::Wrap<T, Self::Base<T, S>> {
        data
    }

    #[inline(always)]
    fn data_as_wrap_mut<'a, T, S: crate::Shape>(
        data: &'a mut Self::Data<T, S>,
    ) -> &'a mut Self::Wrap<T, Self::Base<T, S>> {
        data
    }
}

impl<T, Mods: OnDropBuffer> Alloc<T> for WebGL<Mods> {
    fn alloc<S: Shape>(&self, len: usize, flag: crate::flag::AllocFlag) -> Self::Base<T, S> {
        todo!()
    }

    fn alloc_from_slice<S: Shape>(&self, data: &[T]) -> Self::Base<T, S>
    where
        T: Clone,
    {
        todo!()
    }
}
