use nuklear_rust::*;
use std::io::{Read, BufReader};
use std::fs::File;

pub struct GuiContext {
    pub ctx: NkContext,
    pub allocator: NkAllocator,
    pub cfg: NkConvertConfig,
    pub font: Box<NkFont>,
    pub font_atlas: NkFontAtlas,
}

impl GuiContext {
    pub fn get_cmd(&mut self) -> NkBuffer {
        NkBuffer::with_size(&mut self.allocator, 64 * 1024)
    }
}

pub fn init_gui() -> GuiContext {
    let mut allocator = NkAllocator::new_vec();
    let mut font_atlas = NkFontAtlas::new(&mut allocator);

    let file_path = "data/fonts/DejaVuSans.ttf";
    let mut buffer = Vec::new();
    let file = File::open(&file_path).expect(&format!(
        "Can't open font file:\nFile: {}",
        file_path
    ));
    BufReader::new(file)
        .read_to_end(&mut buffer)
        .expect(&format!(
            "Can't read font file:\nFile: {}",
            file_path
        ));

    let mut cfg = NkFontConfig::with_size(18.0);
    cfg.set_oversample_h(3);
    cfg.set_oversample_v(2);
    cfg.set_glyph_range(font_default_glyph_ranges());
    cfg.set_ttf(include_bytes!("../data/fonts/Roboto-Regular.ttf"));
    font_atlas.begin();
    //let font = font_atlas.add_font_with_config(&cfg).unwrap();
    let font = font_atlas.add_font_with_bytes(include_bytes!("../data/fonts/Roboto-Regular.ttf"), 16.0).unwrap();

    let ctx = NkContext::new(&mut allocator, &font.handle());
    let mut cfg = NkConvertConfig::default();
    let null = NkDrawNullTexture::default();
    cfg.set_null(null.clone());
    cfg.set_circle_segment_count(22);
    cfg.set_curve_segment_count(22);
    cfg.set_arc_segment_count(22);
    cfg.set_global_alpha(1.0);
    cfg.set_shape_aa(NkAntiAliasing::NK_ANTI_ALIASING_ON);
    cfg.set_line_aa(NkAntiAliasing::NK_ANTI_ALIASING_ON);
    GuiContext {
        ctx,
        allocator,
        cfg,
        font,
        font_atlas,
    }
}

pub fn draw_hello_world() {

}
