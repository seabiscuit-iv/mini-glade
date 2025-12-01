use std::sync::Arc;

use nalgebra::Vector3;
use winit::{dpi::PhysicalPosition, event_loop::ActiveEventLoop, keyboard::KeyCode, window::Window};
#[cfg(target_arch = "wasm32")]
use winit::event_loop::{self};

use wgpu::{util::{BufferInitDescriptor, DeviceExt}, wgt::TextureViewDescriptor, *};

use crate::camera::*;
use crate::texture::Texture;
use crate::shader_structs::*;
use crate::helper::*;

pub struct State {
    surface: Surface<'static>,          // the render target essentially
    device: Device,                     // the GPU
    queue: Queue,                       // the work queue for submitting commands to the GPU
    config: SurfaceConfiguration,       // the surface settings
    brown_render_pipeline: RenderPipeline,    // render pipeline handle
    barycentric_render_pipeline: RenderPipeline,    // render pipeline handle
    vertex_buffer: Buffer,
    index_buffer: Buffer,

    #[allow(unused)]
    albedo_texture: Texture,
    albedo_texture_bind_group: BindGroup,

    camera: Camera,
    camera_buffer: Buffer,
    camera_bind_group: BindGroup,

    is_surface_configured: bool,
    triangle_toggle: bool,
    num_indices: u32,

    pub window: Arc<Window>,
    mouse_pos: (f64, f64)
}

impl State {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let (surface, config, device, queue) = configure_surface(window.clone()).await?;

        let diffuse_bytes = include_bytes!("../happy-tree.png");
        let texture = Texture::from_bytes(&device, &queue, diffuse_bytes, "Happy Tree Texture").unwrap();
        let (texture_bind_group_layout, texture_bind_group) = Texture::bind_default_texture(&device, &texture);

        let camera = Camera::from_dimensions(config.width, config.height);
        let camera_uniform = camera.get_uniform();
        let (camera_buffer, camera_bind_group_layout, camera_bind_group) = CameraUniform::bind_camera(&camera_uniform, &device);
                
        let vertex_buffer = device.create_buffer_init(
            &BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(VERTICES),
                usage: BufferUsages::VERTEX
            }
        );

        let index_buffer = device.create_buffer_init(
            &BufferInitDescriptor { 
                label: Some("Index Buffer"), 
                contents: bytemuck::cast_slice(INDICES), 
                usage: BufferUsages::INDEX
            }
        );
        
        let render_pipeline_layout  = device.create_pipeline_layout(
            &PipelineLayoutDescriptor { 
                label: Some("Render Pipeline Layout"), 
                bind_group_layouts: &[&texture_bind_group_layout, &camera_bind_group_layout], 
                push_constant_ranges: &[] 
            }
        );


        let brown_triangle_shader = device.create_shader_module(include_wgsl!("shader.wgsl"));
        let barycentric_triangle_shader = device.create_shader_module(include_wgsl!("barycentric.wgsl"));

        let brown_render_pipeline = make_pipeline_desc_from_shader(&device, &render_pipeline_layout, &brown_triangle_shader, config.format);
        let barycentric_render_pipeline = make_pipeline_desc_from_shader(&device, &render_pipeline_layout, &barycentric_triangle_shader, config.format);

        Ok(Self {
            surface,
            window,
            device,
            queue,
            config,
            is_surface_configured: false,
            brown_render_pipeline,
            barycentric_render_pipeline,
            vertex_buffer,
            mouse_pos: (0.0, 0.0),
            triangle_toggle: true,
            num_indices: INDICES.len() as u32,
            index_buffer,
            albedo_texture_bind_group: texture_bind_group,
            albedo_texture: texture,
            camera,
            camera_bind_group,
            camera_buffer
        })
    }






    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            #[cfg(not(target_arch = "wasm32"))]
            {
                self.config.width = width;
                self.config.height = height;
            }
            #[cfg(target_arch = "wasm32")]
            {
                self.config.width = width;
                self.config.height = height;
            }
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;
            self.camera.aspect_ratio = self.config.width as f32 / self.config.height as f32;
        }
    }






    pub fn handle_key(&mut self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        match (code, is_pressed) {
            (KeyCode::Escape, true) => event_loop.exit(),
            (KeyCode::Space, true) => self.triangle_toggle = !self.triangle_toggle,

            (KeyCode::KeyQ, x) => self.camera.cam_controller.q = x,
            (KeyCode::KeyE, x) => self.camera.cam_controller.e = x,
            (KeyCode::KeyA, x) => self.camera.cam_controller.a = x,
            (KeyCode::KeyD, x) => self.camera.cam_controller.d = x,
            (KeyCode::KeyW, x) => self.camera.cam_controller.w = x,
            (KeyCode::KeyS, x) => self.camera.cam_controller.s = x,

            _ => ()
        }
    }


     
    pub fn handle_mouse_moved(&mut self, _event_loop: &ActiveEventLoop, pos: PhysicalPosition<f64>) {
        self.mouse_pos = (pos.x, pos.y);
    }





    pub fn update(&mut self) {
        self.camera.update();
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[self.camera.get_uniform()]));
    }



    

    pub fn render(&mut self) -> Result<(), SurfaceError>{
        self.window.request_redraw();

        if !self.is_surface_configured {
            return Ok(());
        }

        let output = self.surface.get_current_texture()?;

        let view = output.texture.create_view(&TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder( &CommandEncoderDescriptor {
            label: Some("Render Encoder")
        });

        with_default_render_pass(&mut encoder, &view, |render_pass| {
            render_pass.set_pipeline(if self.triangle_toggle { &self.brown_render_pipeline } else { &self.barycentric_render_pipeline });
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), IndexFormat::Uint16);
            render_pass.set_bind_group(0, &self.albedo_texture_bind_group, &[]);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        });

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

