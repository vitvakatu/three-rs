//! The renderer.

use cgmath::{Matrix4, SquareMatrix, Transform as Transform_, Vector3};
use froggy;
use gfx;
use gfx::handle as h;
use gfx::memory::Typed;
use gfx::traits::{Device, Factory as Factory_, FactoryExt};
use gfx::pso::buffer::Structure;
#[cfg(feature = "opengl")]
use gfx_device_gl as back;
#[cfg(feature = "opengl")]
use gfx_window_glutin;
#[cfg(feature = "opengl")]
use glutin;
use mint;
use nuklear_rust as nk;
use gui;

pub mod source;
mod pso_data;

use color;

use std::{io, str};
use std::mem;
use std::collections::HashMap;
use std::path::PathBuf;

pub use self::back::CommandBuffer as BackendCommandBuffer;
pub use self::back::Factory as BackendFactory;
pub use self::back::Resources as BackendResources;
pub use self::source::Source;

use self::pso_data::PsoData;
use camera::Camera;
use factory::Factory;
use hub::{SubLight, SubNode};
use light::{ShadowMap, ShadowProjection};
use material::Material;
use scene::{Background, Scene};
use text::Font;
use texture::Texture;

/// The format of the back buffer color requested from the windowing system.
pub type ColorFormat = gfx::format::Rgba8;
/// The format of the depth stencil buffer requested from the windowing system.
pub type DepthFormat = gfx::format::DepthStencil;
/// The format of the shadow buffer.
pub type ShadowFormat = gfx::format::Depth32F;
/// The concrete type of a basic pipeline.
pub type BasicPipelineState = gfx::PipelineState<back::Resources, basic_pipe::Meta>;

const MAX_LIGHTS: usize = 4;

const STENCIL_SIDE: gfx::state::StencilSide = gfx::state::StencilSide {
    fun: gfx::state::Comparison::Always,
    mask_read: 0,
    mask_write: 0,
    op_fail: gfx::state::StencilOp::Keep,
    op_depth_fail: gfx::state::StencilOp::Keep,
    op_pass: gfx::state::StencilOp::Keep,
};

#[cfg_attr(rustfmt, rustfmt_skip)]
quick_error! {
    #[doc = "Error encountered when building pipelines."]
    #[derive(Debug)]
    pub enum PipelineCreationError {
        #[doc = "GLSL compiler/linker error."]
        Compilation(err: gfx::shade::ProgramError) {
            from()
            description("GLSL program compilation error")
            display("GLSL program compilation error")
            cause(err)
        }

        #[doc = "Pipeline state error."]
        State(err: gfx::PipelineStateError<String>) {
            from()
            description("Pipeline state error")
            display("Pipeline state error")
            cause(err)
        }

        #[doc = "Standard I/O error."]
        Io(err: io::Error) {
            from()
            description("I/O error")
            display("I/O error")
            cause(err)
        }
    }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
gfx_defines! {
    vertex Vertex {
        pos: [f32; 4] = "a_Position",
        uv: [f32; 2] = "a_TexCoord",
        normal: [gfx::format::I8Norm; 4] = "a_Normal",
        tangent: [gfx::format::I8Norm; 4] = "a_Tangent",
    }

    vertex GuiVertex {
        pos: [f32; 2] = "Position",
        tex: [f32; 2] = "TexCoord",
        col: [gfx::format::U8Norm; 4] = "Color",
    }

    vertex Instance {
        world0: [f32; 4] = "i_World0",
        world1: [f32; 4] = "i_World1",
        world2: [f32; 4] = "i_World2",
        color: [f32; 4] = "i_Color",
        mat_params: [f32; 4] = "i_MatParams",
        uv_range: [f32; 4] = "i_UvRange",
    }

    constant LightParam {
        projection: [[f32; 4]; 4] = "projection",
        pos: [f32; 4] = "pos",
        dir: [f32; 4] = "dir",
        focus: [f32; 4] = "focus",
        color: [f32; 4] = "color",
        color_back: [f32; 4] = "color_back",
        intensity: [f32; 4] = "intensity",
        shadow_params: [i32; 4] = "shadow_params",
    }

    constant Globals {
        mx_vp: [[f32; 4]; 4] = "u_ViewProj",
        mx_inv_proj: [[f32; 4]; 4] = "u_InverseProj",
        mx_view: [[f32; 4]; 4] = "u_View",
        num_lights: u32 = "u_NumLights",
    }

    pipeline basic_pipe {
        vbuf: gfx::VertexBuffer<Vertex> = (),
        inst_buf: gfx::InstanceBuffer<Instance> = (),
        cb_lights: gfx::ConstantBuffer<LightParam> = "b_Lights",
        cb_globals: gfx::ConstantBuffer<Globals> = "b_Globals",
        tex_map: gfx::TextureSampler<[f32; 4]> = "t_Map",
        shadow_map0: gfx::TextureSampler<f32> = "t_Shadow0",
        shadow_map1: gfx::TextureSampler<f32> = "t_Shadow1",
        out_color: gfx::BlendTarget<ColorFormat> =
            ("Target0", gfx::state::MASK_ALL, gfx::preset::blend::REPLACE),
        out_depth: gfx::DepthStencilTarget<DepthFormat> =
            (gfx::preset::depth::LESS_EQUAL_WRITE, gfx::state::Stencil {
                front: STENCIL_SIDE, back: STENCIL_SIDE,
            }),
    }

    pipeline shadow_pipe {
        vbuf: gfx::VertexBuffer<Vertex> = (),
        inst_buf: gfx::InstanceBuffer<Instance> = (),
        cb_globals: gfx::ConstantBuffer<Globals> = "b_Globals",
        target: gfx::DepthTarget<ShadowFormat> =
            gfx::preset::depth::LESS_EQUAL_WRITE,
    }

    constant QuadParams {
        rect: [f32; 4] = "u_Rect",
        depth: f32 = "u_Depth",
    }

    pipeline quad_pipe {
        params: gfx::ConstantBuffer<QuadParams> = "b_Params",
        globals: gfx::ConstantBuffer<Globals> = "b_Globals",
        resource: gfx::RawShaderResource = "t_Input",
        sampler: gfx::Sampler = "t_Input",
        target: gfx::RenderTarget<ColorFormat> = "Target0",
        depth_target: gfx::DepthTarget<DepthFormat> =
            gfx::preset::depth::LESS_EQUAL_TEST,
    }

    constant GuiLocals {
        proj: [[f32; 4]; 4] = "ProjMtx",
    }

    pipeline gui_pipe {
        vbuf: gfx::VertexBuffer<GuiVertex> = (),
        tex: gfx::TextureSampler<[f32; 4]> = "Texture",
        out_color: gfx::BlendTarget<ColorFormat> =
            ("Target0", gfx::state::MASK_ALL, gfx::preset::blend::ALPHA),
        locals: gfx::ConstantBuffer<GuiLocals> = "Locals",
        scissors: gfx::Scissor = (),
    }

    constant PbrParams {
        base_color_factor: [f32; 4] = "u_BaseColorFactor",
        camera: [f32; 3] = "u_Camera",
        _padding0: f32 = "_padding0",
        emissive_factor: [f32; 3] = "u_EmissiveFactor",
        _padding1: f32 = "_padding1",
        metallic_roughness: [f32; 2] = "u_MetallicRoughnessValues",
        normal_scale: f32 = "u_NormalScale",
        occlusion_strength: f32 = "u_OcclusionStrength",
        pbr_flags: i32 = "u_PbrFlags",
    }

    pipeline pbr_pipe {
        vbuf: gfx::VertexBuffer<Vertex> = (),
        inst_buf: gfx::InstanceBuffer<Instance> = (),

        globals: gfx::ConstantBuffer<Globals> = "b_Globals",
        params: gfx::ConstantBuffer<PbrParams> = "b_PbrParams",
        lights: gfx::ConstantBuffer<LightParam> = "b_Lights",

        base_color_map: gfx::TextureSampler<[f32; 4]> = "u_BaseColorSampler",

        normal_map: gfx::TextureSampler<[f32; 4]> = "u_NormalSampler",

        emissive_map: gfx::TextureSampler<[f32; 4]> = "u_EmissiveSampler",

        metallic_roughness_map: gfx::TextureSampler<[f32; 4]> = "u_MetallicRoughnessSampler",

        occlusion_map: gfx::TextureSampler<[f32; 4]> = "u_OcclusionSampler",

        color_target: gfx::RenderTarget<ColorFormat> = "Target0",
        depth_target: gfx::DepthTarget<DepthFormat> = gfx::preset::depth::LESS_EQUAL_WRITE,
    }
}

impl Default for GuiVertex {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct InstanceCacheKey {
    pub(crate) material: Material,
    pub(crate) geometry: h::Buffer<back::Resources, Vertex>,
}

impl Instance {
    #[inline]
    fn basic(
        mx_world: mint::RowMatrix4<f32>,
        color: u32,
        uv_range: [f32; 4],
        param: f32,
    ) -> Self {
        Instance {
            world0: mx_world.x.into(),
            world1: mx_world.y.into(),
            world2: mx_world.z.into(),
            color: {
                // TODO: add alpha parameter for `to_linear_rgb`
                let rgb = color::to_linear_rgb(color);
                [rgb[0], rgb[1], rgb[2], 0.0]
            },
            mat_params: [param, 0.0, 0.0, 0.0],
            uv_range,
        }
    }

    #[inline]
    fn pbr(mx_world: mint::RowMatrix4<f32>) -> Self {
        Instance {
            world0: mx_world.x.into(),
            world1: mx_world.y.into(),
            world2: mx_world.z.into(),
            color: [0.0; 4],
            mat_params: [0.0; 4],
            uv_range: [0.0; 4],
        }
    }
}

//TODO: private fields?
#[derive(Clone, Debug)]
pub(crate) struct GpuData {
    pub slice: gfx::Slice<back::Resources>,
    pub vertices: h::Buffer<back::Resources, Vertex>,
    pub instances: h::Buffer<back::Resources, Instance>,
    pub pending: Option<DynamicData>,
    pub instance_cache_key: Option<InstanceCacheKey>,
}

#[derive(Clone, Debug)]
struct InstanceData {
    pub slice: gfx::Slice<back::Resources>,
    pub vertices: h::Buffer<back::Resources, Vertex>,
    pub pso_data: PsoData,
    pub material: Material,
}

#[derive(Clone, Debug)]
pub(crate) struct DynamicData {
    pub num_vertices: usize,
    pub buffer: h::Buffer<back::Resources, Vertex>,
}

/// Shadow type is used to specify shadow's rendering algorithm.
pub enum ShadowType {
    /// Force no shadows.
    Off,
    /// Basic (and fast) single-sample shadows.
    Basic,
    /// Percentage-closest filter (PCF).
    Pcf,
}

struct DebugQuad {
    resource: h::RawShaderResourceView<back::Resources>,
    pos: [i32; 2],
    size: [i32; 2],
}

/// All pipeline state objects used by the `three` renderer.
pub struct PipelineStates {
    /// Corresponds to `Material::Basic`.
    mesh_basic_fill: BasicPipelineState,

    /// Corresponds to `Material::Line`.
    line_basic: BasicPipelineState,

    /// Corresponds to `Material::Wireframe`.
    mesh_basic_wireframe: BasicPipelineState,

    /// Corresponds to `Material::Gouraud`.
    mesh_gouraud: BasicPipelineState,

    /// Corresponds to `Material::Phong`.
    mesh_phong: BasicPipelineState,

    /// Corresponds to `Material::Sprite`.
    sprite: BasicPipelineState,

    /// Used internally for shadow casting.
    shadow: gfx::PipelineState<back::Resources, shadow_pipe::Meta>,

    /// Used internally for rendering sprites.
    quad: gfx::PipelineState<back::Resources, quad_pipe::Meta>,

    /// Corresponds to `Material::Pbr`.
    pbr: gfx::PipelineState<back::Resources, pbr_pipe::Meta>,

    /// Used internally for rendering `Background::Skybox`.
    skybox: gfx::PipelineState<back::Resources, quad_pipe::Meta>,

    gui: gfx::PipelineState<back::Resources, gui_pipe::Meta>,
}

impl PipelineStates {
    /// Creates the set of pipeline states needed by the `three` renderer.
    pub fn new(
        src: &source::Set,
        factory: &mut Factory,
    ) -> Result<Self, PipelineCreationError> {
        Self::init(src, &mut factory.backend)
    }

    /// Implementation of `PipelineStates::new`.
    pub(crate) fn init(
        src: &source::Set,
        backend: &mut back::Factory,
    ) -> Result<Self, PipelineCreationError> {
        let basic = backend.create_shader_set(&src.basic.vs, &src.basic.ps)?;
        let gouraud = backend.create_shader_set(&src.gouraud.vs, &src.gouraud.ps)?;
        let phong = backend.create_shader_set(&src.phong.vs, &src.phong.ps)?;
        let sprite = backend.create_shader_set(&src.sprite.vs, &src.sprite.ps)?;
        let shadow = backend.create_shader_set(&src.shadow.vs, &src.shadow.ps)?;
        let quad = backend.create_shader_set(&src.quad.vs, &src.quad.ps)?;
        let pbr = backend.create_shader_set(&src.pbr.vs, &src.pbr.ps)?;
        let skybox = backend.create_shader_set(&src.skybox.vs, &src.skybox.ps)?;
        let gui = backend.create_shader_set(&src.gui.vs, &src.gui.ps)?;

        let rast_quad = gfx::state::Rasterizer {
            samples: Some(gfx::state::MultiSample),
            ..gfx::state::Rasterizer::new_fill()
        };
        let rast_fill = rast_quad.with_cull_back();
        let rast_wire = gfx::state::Rasterizer {
            method: gfx::state::RasterMethod::Line(1),
            ..rast_fill
        };
        let rast_shadow = gfx::state::Rasterizer {
            offset: Some(gfx::state::Offset(2, 2)),
            ..rast_fill
        };

        let pso_mesh_basic_fill = backend.create_pipeline_state(
            &basic,
            gfx::Primitive::TriangleList,
            rast_fill,
            basic_pipe::new(),
        )?;
        let pso_line_basic = backend.create_pipeline_state(
            &basic,
            gfx::Primitive::LineStrip,
            rast_fill,
            basic_pipe::new(),
        )?;
        let pso_mesh_basic_wireframe = backend.create_pipeline_state(
            &basic,
            gfx::Primitive::TriangleList,
            rast_wire,
            basic_pipe::new(),
        )?;
        let pso_mesh_gouraud = backend.create_pipeline_state(
            &gouraud,
            gfx::Primitive::TriangleList,
            rast_fill,
            basic_pipe::new(),
        )?;
        let pso_mesh_phong = backend.create_pipeline_state(
            &phong,
            gfx::Primitive::TriangleList,
            rast_fill,
            basic_pipe::new(),
        )?;
        let pso_sprite = backend.create_pipeline_state(
            &sprite,
            gfx::Primitive::TriangleStrip,
            rast_fill,
            basic_pipe::Init {
                out_color: ("Target0", gfx::state::MASK_ALL, gfx::preset::blend::ALPHA),
                ..basic_pipe::new()
            },
        )?;
        let pso_shadow = backend.create_pipeline_state(
            &shadow,
            gfx::Primitive::TriangleList,
            rast_shadow,
            shadow_pipe::new(),
        )?;
        let pso_quad = backend.create_pipeline_state(
            &quad,
            gfx::Primitive::TriangleStrip,
            rast_quad,
            quad_pipe::new(),
        )?;
        let pso_skybox = backend.create_pipeline_state(
            &skybox,
            gfx::Primitive::TriangleStrip,
            rast_quad,
            quad_pipe::new(),
        )?;
        let pso_pbr = backend.create_pipeline_state(
            &pbr,
            gfx::Primitive::TriangleList,
            rast_fill,
            pbr_pipe::new(),
        )?;
        let pso_gui = backend.create_pipeline_state(
            &gui,
            gfx::Primitive::TriangleStrip,
            rast_fill,
            gui_pipe::new(),
        )?;

        Ok(PipelineStates {
            mesh_basic_fill: pso_mesh_basic_fill,
            line_basic: pso_line_basic,
            mesh_basic_wireframe: pso_mesh_basic_wireframe,
            mesh_gouraud: pso_mesh_gouraud,
            mesh_phong: pso_mesh_phong,
            sprite: pso_sprite,
            shadow: pso_shadow,
            quad: pso_quad,
            pbr: pso_pbr,
            skybox: pso_skybox,
            gui: pso_gui,
        })
    }

    pub(crate) fn pso_by_material(
        &self,
        material: &Material,
    ) -> &BasicPipelineState {
        match *material {
            Material::Basic(_) => &self.mesh_basic_fill,
            Material::Line(_) => &self.line_basic,
            Material::Wireframe(_) => &self.mesh_basic_wireframe,
            Material::Lambert(_) => &self.mesh_gouraud,
            Material::Phong(_) => &self.mesh_phong,
            Material::Sprite(_) => &self.sprite,
            _ => unreachable!(),
        }
    }
}

/// Handle for additional viewport to render some relevant debug information.
/// See [`Renderer::debug_shadow_quad`](struct.Renderer.html#method.debug_shadow_quad).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DebugQuadHandle(froggy::Pointer<DebugQuad>);

/// Renders [`Scene`](struct.Scene.html) by [`Camera`](struct.Camera.html).
///
/// See [Window::render](struct.Window.html#method.render).
pub struct Renderer {
    device: back::Device,
    encoder: gfx::Encoder<back::Resources, back::CommandBuffer>,
    factory: back::Factory,
    const_buf: h::Buffer<back::Resources, Globals>,
    quad_buf: h::Buffer<back::Resources, QuadParams>,
    inst_buf: h::Buffer<back::Resources, Instance>,
    light_buf: h::Buffer<back::Resources, LightParam>,
    pbr_buf: h::Buffer<back::Resources, PbrParams>,
    out_color: h::RenderTargetView<back::Resources, ColorFormat>,
    out_depth: h::DepthStencilView<back::Resources, DepthFormat>,
    pso: PipelineStates,
    map_default: Texture<[f32; 4]>,
    shadow_default: Texture<f32>,
    debug_quads: froggy::Storage<DebugQuad>,
    size: (u32, u32),
    font_cache: HashMap<PathBuf, Font>,
    instance_cache: HashMap<InstanceCacheKey, (InstanceData, Vec<Instance>)>,
    /// `ShadowType` of this `Renderer`.
    pub shadow: ShadowType,
    // GUI
    ctx: gui::GuiContext,
    cmd: nk::NkBuffer,
    textures: Vec<Texture<[f32; 4]>>,
    vbf: h::Buffer<back::Resources, GuiVertex>,
    ebf: h::Buffer<back::Resources, u16>,
    lbf: h::Buffer<back::Resources, GuiLocals>,
    vsz: usize,
    esz: usize,
    vle: nk::NkDrawVertexLayoutElements,
}

impl Renderer {
    #[cfg(feature = "opengl")]
    pub(crate) fn new(
        builder: glutin::WindowBuilder,
        context: glutin::ContextBuilder,
        event_loop: &glutin::EventsLoop,
        source: &source::Set,
    ) -> (Self, glutin::GlWindow, Factory) {
        use gfx::texture as t;
        let (window, device, mut gl_factory, out_color, out_depth) = gfx_window_glutin::init(builder, context, event_loop);
        let (_, srv_white) = gl_factory
            .create_texture_immutable::<gfx::format::Rgba8>(t::Kind::D2(1, 1, t::AaMode::Single), &[&[[0xFF; 4]]])
            .unwrap();
        let (_, srv_shadow) = gl_factory
            .create_texture_immutable::<(gfx::format::R32, gfx::format::Float)>(t::Kind::D2(1, 1, t::AaMode::Single), &[&[0x3F800000]])
            .unwrap();
        let sampler = gl_factory.create_sampler_linear();
        let sampler_shadow = gl_factory.create_sampler(t::SamplerInfo {
            comparison: Some(gfx::state::Comparison::Less),
            border: t::PackedColor(!0), // clamp to 1.0
            ..t::SamplerInfo::new(t::FilterMethod::Bilinear, t::WrapMode::Border)
        });
        let encoder = gl_factory.create_command_buffer().into();
        let const_buf = gl_factory.create_constant_buffer(1);
        let quad_buf = gl_factory.create_constant_buffer(1);
        let light_buf = gl_factory.create_constant_buffer(MAX_LIGHTS);
        let pbr_buf = gl_factory.create_constant_buffer(1);
        let inst_buf = gl_factory
            .create_buffer(
                1,
                gfx::buffer::Role::Vertex,
                gfx::memory::Usage::Dynamic,
                gfx::TRANSFER_DST,
            )
            .unwrap();
        let pso = PipelineStates::init(source, &mut gl_factory).unwrap();

        let mut allocator = nk::NkAllocator::new_vec();
        let mut font_atlas = nk::NkFontAtlas::new(&mut allocator);

        let mut cfg = nk::NkFontConfig::with_size(28.0);
        cfg.set_oversample_h(3);
        cfg.set_oversample_v(2);
        cfg.set_glyph_range(nk::font_cyrillic_glyph_ranges());
        cfg.set_ttf(include_bytes!("../../data/fonts/DejaVuSans.ttf"));
        font_atlas.begin();
        let font = font_atlas.add_font_with_config(&cfg).unwrap();

        let ctx = nk::NkContext::new(&mut allocator, &font.handle());

        let mut cfg = nk::NkConvertConfig::default();
        let mut null = nk::NkDrawNullTexture::default();
        //cfg.set_null(null.clone());
        cfg.set_circle_segment_count(22);
        cfg.set_curve_segment_count(22);
        cfg.set_arc_segment_count(22);
        cfg.set_global_alpha(1.0);
        cfg.set_shape_aa(nk::NkAntiAliasing::NK_ANTI_ALIASING_ON);
        cfg.set_line_aa(nk::NkAntiAliasing::NK_ANTI_ALIASING_ON);

        let mut gui_ctx = gui::GuiContext {
            ctx,
            allocator,
            cfg,
            font,
            font_atlas,
        };

        println!("Font count: {}", gui_ctx.font_atlas.fonts().len());

        let mut textures = Vec::new();
        let font_tex = {
            let (buf, width, height) = gui_ctx.font_atlas.bake(nk::NkFontAtlasFormat::NK_FONT_ATLAS_RGBA32);
            println!("Font_texture: {} {} {}", buf.len(), width, height);
            use image;
            image::save_buffer("/home/fumlead/projects/three-rs/test.png", buf, width, height, image::ColorType::RGBA(8)).unwrap();
            use gfx::texture as t;
            let kind = t::Kind::D2(width as u16, height as u16, t::AaMode::Single);
            let (_, view) = gl_factory
                .create_texture_immutable_u8::<gfx::format::Rgba8>(kind, &[buf])
                .unwrap_or_else(|e| {
                    panic!("Unable to create GPU texture from memory: {:?}", e);
                });
            let texture = Texture::new(view, sampler.clone(), [width as u32, height as u32]);
            textures.push(texture);
            nk::NkHandle::from_id(textures.len() as i32)
        };
        //gui_ctx.font_atlas.end(font_tex, Some(&mut null));
        gui_ctx.font_atlas.end(font_tex, None);

        let cmd = gui_ctx.get_cmd();
        let renderer = Renderer {
            device,
            factory: gl_factory.clone(),
            encoder,
            const_buf,
            quad_buf,
            light_buf,
            inst_buf,
            pbr_buf,
            out_color,
            out_depth,
            pso,
            map_default: Texture::new(srv_white, sampler, [1, 1]),
            shadow_default: Texture::new(srv_shadow, sampler_shadow, [1, 1]),
            instance_cache: HashMap::new(),
            shadow: ShadowType::Basic,
            debug_quads: froggy::Storage::new(),
            font_cache: HashMap::new(),
            size: window.get_inner_size_pixels().unwrap(),
            // GUI
            ctx: gui_ctx,
            cmd,
            textures,
            vbf: gl_factory.create_upload_buffer::<GuiVertex>(512 * 1024).unwrap(),
            ebf: gl_factory.create_upload_buffer::<u16>(128 * 1024).unwrap(),
            vsz: 512 * 1024,
            esz: 128 * 1024,
            lbf: gl_factory.create_constant_buffer::<GuiLocals>(1),
            vle: nk::NkDrawVertexLayoutElements::new(&[
                (nk::NkDrawVertexLayoutAttribute::NK_VERTEX_POSITION, nk::NkDrawVertexLayoutFormat::NK_FORMAT_FLOAT, GuiVertex::query("Position").unwrap().offset),
                (nk::NkDrawVertexLayoutAttribute::NK_VERTEX_TEXCOORD, nk::NkDrawVertexLayoutFormat::NK_FORMAT_FLOAT, GuiVertex::query("TexCoord").unwrap().offset),
                (nk::NkDrawVertexLayoutAttribute::NK_VERTEX_COLOR, nk::NkDrawVertexLayoutFormat::NK_FORMAT_R8G8B8A8, GuiVertex::query("Color").unwrap().offset),
                (nk::NkDrawVertexLayoutAttribute::NK_VERTEX_ATTRIBUTE_COUNT, nk::NkDrawVertexLayoutFormat::NK_FORMAT_COUNT, 0u32)
            ]),
        };
        let factory = Factory::new(gl_factory);
        (renderer, window, factory)
    }

    pub fn add_texture(&mut self, texture: Texture<[f32; 4]>) -> nk::NkHandle {
        self.textures.push(texture);
        nk::NkHandle::from_id(self.textures.len() as i32)
    }

    fn find_texture(&self, id: i32) -> Option<Texture<[f32; 4]>> {
        let mut ret = None;

        if id > 0 && id as usize <= self.textures.len() {
            ret = Some(self.textures[(id - 1) as usize].clone())
        }
        ret
    }

    /// Reloads the shaders.
    pub fn reload(
        &mut self,
        pipeline_states: PipelineStates,
    ) {
        self.pso = pipeline_states;
    }

    pub(crate) fn resize(
        &mut self,
        window: &glutin::GlWindow,
    ) {
        let size = window.get_inner_size_pixels().unwrap();

        // skip updating view and self size if some
        // of the sides equals to zero (fixes crash on minimize on Windows machines)
        if size.0 == 0 || size.1 == 0 {
            return;
        }

        self.size = size;
        gfx_window_glutin::update_views(window, &mut self.out_color, &mut self.out_depth);
    }

    /// Returns current viewport aspect ratio, i.e. width / height.
    pub fn aspect_ratio(&self) -> f32 {
        self.size.0 as f32 / self.size.1 as f32
    }

    /// Map screen pixel coordinates to Normalized Display Coordinates.
    /// The lower left corner corresponds to (-1,-1), and the upper right corner
    /// corresponds to (1,1).
    pub fn map_to_ndc<P: Into<mint::Point2<f32>>>(
        &self,
        point: P,
    ) -> mint::Point2<f32> {
        let point = point.into();
        mint::Point2 {
            x: 2.0 * point.x / self.size.0 as f32 - 1.0,
            y: 1.0 - 2.0 * point.y / self.size.1 as f32,
        }
    }

    /// See [`Window::render`](struct.Window.html#method.render).
    pub fn render(
        &mut self,
        scene: &Scene,
        camera: &Camera,
    ) {
        let mut hub = scene.hub.lock().unwrap();
        hub.process_messages();
        self.device.cleanup();
        self.ctx.ctx.clear();

        // update dynamic meshes
        // Note: mutable node access here
        for node in hub.nodes.iter_mut() {
            if !node.visible {
                continue;
            }
            match node.sub_node {
                SubNode::Visual(_, ref mut gpu_data) => {
                    if let Some(dynamic) = gpu_data.pending.take() {
                        self.encoder
                            .copy_buffer(
                                &dynamic.buffer,
                                &gpu_data.vertices,
                                0,
                                0,
                                dynamic.num_vertices,
                            )
                            .unwrap();
                    }
                }
                // Note: UI text currently applies to all the scenes.
                // We may want to make it scene-dependent at some point.
                SubNode::UiText(ref text) => {
                    text.font.queue(&text.section);
                    if !self.font_cache.contains_key(&text.font.path) {
                        self.font_cache
                            .insert(text.font.path.clone(), text.font.clone());
                    }
                }
                _ => {}
            }
        }

        // gather lights
        struct ShadowRequest {
            target: h::DepthStencilView<back::Resources, ShadowFormat>,
            resource: h::ShaderResourceView<back::Resources, f32>,
            mx_view: Matrix4<f32>,
            mx_proj: Matrix4<f32>,
        }
        let mut lights = Vec::new();
        let mut shadow_requests = Vec::new();
        let mut mx_camera_transform = hub[&camera].transform;

        for w in hub.walk(&scene.first_child) {
            // grab the camera world space info
            if w.node as *const _ == &hub[&camera] as *const _ {
                mx_camera_transform = w.world_transform;
            }
            let light = match w.node.sub_node {
                SubNode::Light(ref light) => light,
                _ => continue,
            };
            if lights.len() == MAX_LIGHTS {
                error!("Max number of lights ({}) reached", MAX_LIGHTS);
                break;
            }

            let shadow_index = if let Some((ref map, ref projection)) = light.shadow {
                let target = map.to_target();
                let dim = target.get_dimensions();
                let aspect = dim.0 as f32 / dim.1 as f32;
                let mx_proj = match projection {
                    &ShadowProjection::Orthographic(ref p) => p.matrix(aspect),
                };
                let mx_view = Matrix4::from(w.world_transform.inverse_transform().unwrap());
                shadow_requests.push(ShadowRequest {
                    target,
                    resource: map.to_resource(),
                    mx_view,
                    mx_proj: mx_proj.into(),
                });
                shadow_requests.len() as i32 - 1
            } else {
                -1
            };

            let mut color_back = 0;
            let mut p = w.world_transform.disp.extend(1.0);
            let d = w.world_transform.rot * Vector3::unit_z();
            let intensity = match light.sub_light {
                SubLight::Ambient => [light.intensity, 0.0, 0.0, 0.0],
                SubLight::Directional => {
                    p = d.extend(0.0);
                    [0.0, light.intensity, 0.0, 0.0]
                }
                SubLight::Hemisphere { ground } => {
                    color_back = ground | 0x010101; // can't be 0
                    p = d.extend(0.0);
                    [light.intensity, 0.0, 0.0, 0.0]
                }
                SubLight::Point => [0.0, light.intensity, 0.0, 0.0],
            };
            let projection = if shadow_index >= 0 {
                let request = &shadow_requests[shadow_index as usize];
                let matrix = request.mx_proj * request.mx_view;
                matrix.into()
            } else {
                [[0.0; 4]; 4]
            };

            lights.push(LightParam {
                projection,
                pos: p.into(),
                dir: d.extend(0.0).into(),
                focus: [0.0, 0.0, 0.0, 0.0],
                color: {
                    let rgb = color::to_linear_rgb(light.color);
                    [rgb[0], rgb[1], rgb[2], 0.0]
                },
                color_back: {
                    let rgb = color::to_linear_rgb(color_back);
                    [rgb[0], rgb[1], rgb[2], 0.0]
                },
                intensity,
                shadow_params: [shadow_index, 0, 0, 0],
            });
        }

        // render shadow maps
        for request in &shadow_requests {
            self.encoder.clear_depth(&request.target, 1.0);
            let mx_vp = request.mx_proj * request.mx_view;
            self.encoder.update_constant_buffer(
                &self.const_buf,
                &Globals {
                    mx_vp: mx_vp.into(),
                    mx_view: request.mx_view.into(),
                    mx_inv_proj: request.mx_proj.into(),
                    num_lights: 0,
                },
            );

            for w in hub.walk(&scene.first_child) {
                let gpu_data = match w.node.sub_node {
                    SubNode::Visual(_, ref data) => data,
                    _ => continue,
                };
                let mx_world: mint::ColumnMatrix4<_> = Matrix4::from(w.world_transform).into();
                self.encoder
                    .update_buffer(&gpu_data.instances, &[Instance::pbr(mx_world.into())], 0)
                    .unwrap();
                //TODO: avoid excessive cloning
                let data = shadow_pipe::Data {
                    vbuf: gpu_data.vertices.clone(),
                    inst_buf: gpu_data.instances.clone(),
                    cb_globals: self.const_buf.clone(),
                    target: request.target.clone(),
                };
                self.encoder.draw(&gpu_data.slice, &self.pso.shadow, &data);
            }
        }

        // prepare target and globals
        let mx_view = Matrix4::from(mx_camera_transform.inverse_transform().unwrap());
        let mx_proj = Matrix4::from(camera.matrix(self.aspect_ratio()));
        self.encoder.update_constant_buffer(
            &self.const_buf,
            &Globals {
                mx_vp: (mx_proj * mx_view).into(),
                mx_view: mx_view.into(),
                mx_inv_proj: mx_proj.invert().unwrap().into(),
                num_lights: lights.len() as u32,
            },
        );
        self.encoder
            .update_buffer(&self.light_buf, &lights, 0)
            .unwrap();

        self.encoder.clear_depth(&self.out_depth, 1.0);
        self.encoder.clear_stencil(&self.out_depth, 0);

        if let Background::Color(color) = scene.background {
            let rgb = color::to_linear_rgb(color);
            self.encoder
                .clear(&self.out_color, [rgb[0], rgb[1], rgb[2], 0.0]);
        }

        // render everything
        let (shadow_default, shadow_sampler) = self.shadow_default.to_param();
        let shadow0 = match shadow_requests.get(0) {
            Some(ref request) => request.resource.clone(),
            None => shadow_default.clone(),
        };
        let shadow1 = match shadow_requests.get(1) {
            Some(ref request) => request.resource.clone(),
            None => shadow_default.clone(),
        };

        // clear instance cache
        for instances in self.instance_cache.values_mut() {
            instances.1.clear();
        }

        for w in hub.walk(&scene.first_child) {
            let (material, gpu_data) = match w.node.sub_node {
                SubNode::Visual(ref mat, ref data) => (mat, data),
                _ => continue,
            };

            let mx_world: mint::ColumnMatrix4<_> = Matrix4::from(w.world_transform).into();
            let pso_data = material.to_pso_data();

            if let Some(ref key) = gpu_data.instance_cache_key {
                let uv_range = [0.0; 4];
                let (color, mat_param) = match pso_data {
                    PsoData::Basic { color, param0, .. } => (color, param0),
                    PsoData::Pbr { .. } => (!0, 0.0),
                };
                let vec = self.instance_cache.entry(key.clone()).or_insert((
                    InstanceData {
                        slice: gpu_data.slice.clone(),
                        vertices: gpu_data.vertices.clone(),
                        pso_data: pso_data.clone(),
                        material: material.clone(),
                    },
                    Vec::new(),
                ));
                vec.1
                    .push(Instance::basic(mx_world.into(), color, uv_range, mat_param));
                continue;
            }
            let instance = match pso_data {
                PsoData::Basic { color, map, param0 } => {
                    let uv_range = match map {
                        Some(ref map) => map.uv_range(),
                        None => [0.0; 4],
                    };
                    Instance::basic(mx_world.into(), color, uv_range, param0)
                }
                PsoData::Pbr { .. } => Instance::pbr(mx_world.into()),
            };

            Self::render_mesh(
                &mut self.encoder,
                self.const_buf.clone(),
                gpu_data.instances.clone(),
                self.light_buf.clone(),
                self.pbr_buf.clone(),
                self.out_color.clone(),
                self.out_depth.clone(),
                &self.pso,
                &self.map_default,
                &[instance],
                gpu_data.vertices.clone(),
                gpu_data.slice.clone(),
                &material,
                &shadow_sampler,
                &shadow0,
                &shadow1,
            );
        }

        // render instanced meshes
        for &(ref mesh_data, ref all_instances) in self.instance_cache.values() {
            if all_instances.len() > self.inst_buf.len() {
                self.inst_buf = self.factory
                    .create_buffer(
                        all_instances.len(),
                        gfx::buffer::Role::Vertex,
                        gfx::memory::Usage::Dynamic,
                        gfx::TRANSFER_DST,
                    )
                    // TODO: Better error handling
                    .unwrap();
            }
            Self::render_mesh(
                &mut self.encoder,
                self.const_buf.clone(),
                self.inst_buf.clone(),
                self.light_buf.clone(),
                self.pbr_buf.clone(),
                self.out_color.clone(),
                self.out_depth.clone(),
                &self.pso,
                &self.map_default,
                all_instances,
                mesh_data.vertices.clone(),
                mesh_data.slice.clone(),
                &mesh_data.material,
                &shadow_sampler,
                &shadow0,
                &shadow1,
            );
        }

        let quad_slice = gfx::Slice {
            start: 0,
            end: 4,
            base_vertex: 0,
            instances: None,
            buffer: gfx::IndexBuffer::Auto,
        };

        // draw background (if any)
        match scene.background {
            Background::Texture(ref texture) => {
                // TODO: Reduce code duplication (see drawing debug quads)
                self.encoder.update_constant_buffer(
                    &self.quad_buf,
                    &QuadParams {
                        rect: [-1.0, -1.0, 1.0, 1.0],
                        depth: 1.0,
                    },
                );
                let data = quad_pipe::Data {
                    params: self.quad_buf.clone(),
                    globals: self.const_buf.clone(),
                    resource: texture.to_param().0.raw().clone(),
                    sampler: texture.to_param().1,
                    target: self.out_color.clone(),
                    depth_target: self.out_depth.clone(),
                };
                self.encoder.draw(&quad_slice, &self.pso.quad, &data);
            }
            Background::Skybox(ref cubemap) => {
                self.encoder.update_constant_buffer(
                    &self.quad_buf,
                    &QuadParams {
                        rect: [-1.0, -1.0, 1.0, 1.0],
                        depth: 1.0,
                    },
                );
                let data = quad_pipe::Data {
                    params: self.quad_buf.clone(),
                    resource: cubemap.to_param().0.raw().clone(),
                    sampler: cubemap.to_param().1,
                    globals: self.const_buf.clone(),
                    target: self.out_color.clone(),
                    depth_target: self.out_depth.clone(),
                };
                self.encoder.draw(&quad_slice, &self.pso.skybox, &data);
            }
            Background::Color(_) => {}
        }

        // draw ui text
        for (_, font) in &self.font_cache {
            font.draw(&mut self.encoder, &self.out_color, &self.out_depth);
        }

        // draw debug quads
        self.debug_quads.sync_pending();
        for quad in self.debug_quads.iter() {
            let pos = [
                if quad.pos[0] >= 0 {
                    quad.pos[0]
                } else {
                    self.size.0 as i32 + quad.pos[0] - quad.size[0]
                },
                if quad.pos[1] >= 0 {
                    quad.pos[1]
                } else {
                    self.size.1 as i32 + quad.pos[1] - quad.size[1]
                },
            ];
            let p0 = self.map_to_ndc([pos[0] as f32, pos[1] as f32]);
            let p1 = self.map_to_ndc([
                (pos[0] + quad.size[0]) as f32,
                (pos[1] + quad.size[1]) as f32,
            ]);
            self.encoder.update_constant_buffer(
                &self.quad_buf,
                &QuadParams {
                    rect: [p0.x, p0.y, p1.x, p1.y],
                    depth: -1.0,
                },
            );
            let data = quad_pipe::Data {
                params: self.quad_buf.clone(),
                globals: self.const_buf.clone(),
                resource: quad.resource.clone(),
                sampler: self.map_default.to_param().1,
                target: self.out_color.clone(),
                depth_target: self.out_depth.clone(),
            };
            self.encoder.draw(&quad_slice, &self.pso.quad, &data);
        }

        //GUI
        self.ctx.ctx.style_set_font(&self.ctx.font.handle());
        if self.ctx.ctx.begin(nk_string!("Nuklear with Three-rs"),
            nk::NkRect {
                x: 100.0,
                y: 100.0,
                w: 300.0,
                h: 300.0,
            },
            nk::NkPanelFlags::NK_WINDOW_BORDER as nk::NkFlags | nk::NkPanelFlags::NK_WINDOW_TITLE as nk::NkFlags) {
            //self.ctx.ctx.layout_row_dynamic(20.0, 1);
            //self.ctx.ctx.text("Nuklear and Three-rs", nk::NkTextAlignment::NK_TEXT_LEFT as nk::NkFlags);
            self.ctx.ctx.end();
        }

        let width = self.size.0;
        let height = self.size.1;
        let scale = nk::NkVec2 {
            x: 1.0,
            y: 1.0,
        };
        use gfx::IntoIndexBuffer;
        let ortho = [[2.0f32 / width as f32, 0.0f32, 0.0f32, 0.0f32],
            [0.0f32, -2.0f32 / height as f32, 0.0f32, 0.0f32],
            [0.0f32, 0.0f32, -1.0f32, 0.0f32],
            [-1.0f32, 1.0f32, 0.0f32, 1.0f32]];

        self.ctx.cfg.set_vertex_layout(&self.vle);
        self.ctx.cfg.set_vertex_size(mem::size_of::<GuiVertex>());
        {
            let mut rwv = self.factory.write_mapping(&mut self.vbf).unwrap();
            let mut rvbuf = unsafe {
                ::std::slice::from_raw_parts_mut(&mut *rwv as *mut [GuiVertex] as *mut u8,
                                                 ::std::mem::size_of::<GuiVertex>() * self.vsz)
            };
            let mut vbuf = nk::NkBuffer::with_fixed(&mut rvbuf);

            let mut rwe = self.factory.write_mapping(&mut self.ebf).unwrap();
            let mut rebuf = unsafe {
                ::std::slice::from_raw_parts_mut(&mut *rwe as *mut [u16] as *mut u8,
                                                 ::std::mem::size_of::<u16>() * self.esz)
            };
            let mut ebuf = nk::NkBuffer::with_fixed(&mut rebuf);

            self.ctx.ctx.convert(&mut self.cmd, &mut vbuf, &mut ebuf, &self.ctx.cfg);
        }
        let mut slice = ::gfx::Slice {
            start: 0,
            end: 0,
            base_vertex: 0,
            instances: None,
            buffer: self.ebf.clone().into_index_buffer(&mut self.factory),
        };

        self.encoder.update_constant_buffer(&mut self.lbf, &GuiLocals { proj: ortho });

        for (i, cmd) in self.ctx.ctx.draw_command_iterator(&self.cmd).into_iter().enumerate() {
            println!("Command #{}: elements: {} texture: {}", i, cmd.elem_count(), cmd.texture().id().unwrap());
            if cmd.elem_count() < 1 {
                continue;
            }

            slice.end = slice.start + cmd.elem_count();

            let id = cmd.texture().id().unwrap();

            let x = cmd.clip_rect().x * scale.x;
            let y = cmd.clip_rect().y * scale.y;
            let w = cmd.clip_rect().w * scale.x;
            let h = cmd.clip_rect().h * scale.y;

            let sc_rect = gfx::Rect {
                x: (if x < 0f32 { 0f32 } else { x }) as u16,
                y: (if y < 0f32 { 0f32 } else { y }) as u16,
                w: (if x < 0f32 { w + x } else { w }) as u16,
                h: (if y < 0f32 { h + y } else { h }) as u16,
            };

            let res = if id > 0 && (id as usize) <= self.textures.len() {
                self.textures[(id - 1) as usize].clone()
            } else {
                println!("Miss {}!", id);
                self.textures[0].clone()
                //self.map_default.clone()
            };

            let data = gui_pipe::Data {
                vbuf: self.vbf.clone(),
                tex: res.to_param(),
                out_color: self.out_color.clone(),
                scissors: sc_rect,
                locals: self.lbf.clone(),
            };

            self.encoder.draw(&slice, &self.pso.gui, &data);

            slice.start = slice.end;
        }

        self.encoder.flush(&mut self.device);
    }

    #[inline]
    fn render_mesh(
        encoder: &mut gfx::Encoder<back::Resources, back::CommandBuffer>,
        const_buf: h::Buffer<back::Resources, Globals>,
        inst_buf: h::Buffer<back::Resources, Instance>,
        light_buf: h::Buffer<back::Resources, LightParam>,
        pbr_buf: h::Buffer<back::Resources, PbrParams>,
        out_color: h::RenderTargetView<back::Resources, ColorFormat>,
        out_depth: h::DepthStencilView<back::Resources, DepthFormat>,
        pso: &PipelineStates,
        map_default: &Texture<[f32; 4]>,
        instances: &[Instance],
        vertex_buf: h::Buffer<back::Resources, Vertex>,
        slice: gfx::Slice<back::Resources>,
        material: &Material,
        shadow_sampler: &h::Sampler<back::Resources>,
        shadow0: &h::ShaderResourceView<back::Resources, f32>,
        shadow1: &h::ShaderResourceView<back::Resources, f32>,
    ) {
        encoder.update_buffer(&inst_buf, instances, 0).unwrap();

        let slice = if instances.len() > 1 {
            gfx::Slice {
                instances: Some((instances.len() as u32, 0)),
                ..slice
            }
        } else {
            slice
        };

        //TODO: batch per PSO
        match material.to_pso_data() {
            PsoData::Pbr { maps, params } => {
                encoder.update_constant_buffer(&pbr_buf, &params);
                let map_params = maps.into_params(map_default);
                let data = pbr_pipe::Data {
                    vbuf: vertex_buf,
                    inst_buf: inst_buf,
                    globals: const_buf.clone(),
                    lights: light_buf.clone(),
                    params: pbr_buf.clone(),
                    base_color_map: map_params.base_color,
                    normal_map: map_params.normal,
                    emissive_map: map_params.emissive,
                    metallic_roughness_map: map_params.metallic_roughness,
                    occlusion_map: map_params.occlusion,
                    color_target: out_color.clone(),
                    depth_target: out_depth.clone(),
                };
                encoder.draw(&slice, &pso.pbr, &data);
            }
            PsoData::Basic { map, .. } => {
                //TODO: avoid excessive cloning
                let data = basic_pipe::Data {
                    vbuf: vertex_buf,
                    inst_buf: inst_buf,
                    cb_lights: light_buf.clone(),
                    cb_globals: const_buf.clone(),
                    tex_map: map.unwrap_or(map_default.clone()).to_param(),
                    shadow_map0: (shadow0.clone(), shadow_sampler.clone()),
                    shadow_map1: (shadow1.clone(), shadow_sampler.clone()),
                    out_color: out_color.clone(),
                    out_depth: (out_depth.clone(), (0, 0)),
                };
                encoder.draw(&slice, pso.pso_by_material(&material), &data);
            }
        }
    }

    /// Draw [`ShadowMap`](struct.ShadowMap.html) for debug purposes.
    pub fn debug_shadow_quad(
        &mut self,
        map: &ShadowMap,
        _num_components: u8,
        pos: [i16; 2],
        size: [u16; 2],
    ) -> DebugQuadHandle {
        DebugQuadHandle(self.debug_quads.create(DebugQuad {
            resource: map.to_resource().raw().clone(),
            pos: [pos[0] as i32, pos[1] as i32],
            size: [size[0] as i32, size[1] as i32],
        }))
    }
}
