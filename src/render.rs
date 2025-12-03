use std::{sync::Arc};

use egui::{Align2, Color32, Margin, ViewportId};
use egui_winit::winit;
use egui_wgpu::wgpu;

use instant::Instant;
use nalgebra::{Matrix4, Vector2, Vector3, Vector4};
use winit::{dpi::PhysicalPosition, event::MouseButton, event_loop::ActiveEventLoop, keyboard::KeyCode, window::Window};
#[cfg(target_arch = "wasm32")]
use winit::event_loop::{self};

use wgpu::{util::{BufferInitDescriptor, DeviceExt}, *};

use crate::camera::*;
use crate::shader_structs::*;
use crate::helper::*;
use crate::scene::*;


pub struct State {
    surface: Surface<'static>,          // the render target essentially
    device: Device,                     // the GPU
    queue: Queue,                       // the work queue for submitting commands to the GPU
    config: SurfaceConfiguration,       // the surface settings
    brown_render_pipeline: RenderPipeline,    // render pipeline handle

    vertex_buffer: Buffer,
    index_buffer: Buffer,

    camera: Camera,
    camera_buffer: Buffer,
    camera_bind_group: BindGroup,

    scene: SceneUniform,
    scene_buffer: Buffer,
    scene_bind_group: BindGroup,

    is_surface_configured: bool,
    num_indices: u32,

    pub window: Arc<Window>,

    mouse_pos: (f64, f64),
    mouse_delta: (f64, f64),

    start: Instant,

    //egui
    pub egui_state: egui_winit::State,
    pub egui_renderer: egui_wgpu::Renderer
}

impl State {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let (surface, config, device, queue) = configure_surface(window.clone()).await?;

        let camera = Camera::from_dimensions(config.width, config.height);
        let camera_uniform = camera.get_uniform();
        let (camera_buffer, camera_bind_group_layout, camera_bind_group) = CameraUniform::bind_camera(&camera_uniform, &device);

        let scene = SceneUniform {
            num_objects: 1,
            selected_object: 999,
            time: 0.0,
            padding_0: [0.0; 1],
            object_positions: [[3.0, 3.0, 0.0, 1.0]; OBJECT_MAX as usize],
            object_rotations: [[0.0; 4]; OBJECT_MAX as usize],
            object_meta: [[0; 4]; OBJECT_MAX as usize],
            object_param_1: [[0.0; 4]; OBJECT_MAX as usize],
            object_param_2: [[0.0; 4]; OBJECT_MAX as usize],
            object_param_3: [[0.0; 4]; OBJECT_MAX as usize],
            object_param_4: [[0.0; 4]; OBJECT_MAX as usize],
        };
        let (scene_buffer, scene_bind_group_layout, scene_bind_group) = bind_scene(&scene, &device);
                
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
                bind_group_layouts: &[&camera_bind_group_layout, &scene_bind_group_layout], 
                push_constant_ranges: &[] 
            }
        );

        let brown_triangle_shader = device.create_shader_module(include_wgsl!("shader.wgsl"));

        let brown_render_pipeline = make_pipeline_desc_from_shader(&device, &render_pipeline_layout, &brown_triangle_shader, config.format);

        let egui_context = egui::Context::default();
        let mut style = (*egui_context.style()).clone();
        style.text_styles = [
                (egui::TextStyle::Heading, egui::FontId::new(13.0, egui::FontFamily::Proportional)),
                (egui::TextStyle::Body,    egui::FontId::new(10.0, egui::FontFamily::Proportional)),
                (egui::TextStyle::Button,  egui::FontId::new(10.0, egui::FontFamily::Proportional)),
                (egui::TextStyle::Monospace, egui::FontId::new(10.0, egui::FontFamily::Monospace)),
            ]
            .into(); 
        style.compact_menu_style = true;
        style.spacing.window_margin = Margin::same(8);
        egui_context.set_style(style);

        let egui_state = egui_winit::State::new(egui_context, ViewportId::ROOT, &window, None, None, None);        

        Ok(Self {
            egui_state,
            egui_renderer: egui_wgpu::Renderer::new(&device, config.format, 
                egui_wgpu::RendererOptions { 
                    msaa_samples: 1, 
                    depth_stencil_format: None, 
                    dithering: false, 
                    predictable_texture_filtering: false
                }
            ),
            surface,
            window,
            device,
            queue,
            config,
            is_surface_configured: false,
            brown_render_pipeline,
            vertex_buffer,
            mouse_pos: (0.0, 0.0),
            mouse_delta: (0.0, 0.0),
            num_indices: INDICES.len() as u32,
            index_buffer,
            camera,
            camera_bind_group,
            camera_buffer,
            scene,
            scene_buffer,
            scene_bind_group,
            start: Instant::now()
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

            (KeyCode::KeyQ, x) => self.camera.cam_controller.q = x,
            (KeyCode::KeyE, x) => self.camera.cam_controller.e = x,
            (KeyCode::KeyA, x) => self.camera.cam_controller.a = x,
            (KeyCode::KeyD, x) => self.camera.cam_controller.d = x,
            (KeyCode::KeyW, x) => self.camera.cam_controller.w = x,
            (KeyCode::KeyS, x) => self.camera.cam_controller.s = x,

            _ => ()
        }
    }




    pub fn handle_mouse_event(&mut self, button: MouseButton, is_pressed: bool) {
        match button {
            MouseButton::Left => self.camera.cam_controller.dragging = is_pressed,
            _ => ()
        }
    }



     
    pub fn handle_mouse_moved(&mut self, _event_loop: &ActiveEventLoop, pos: PhysicalPosition<f64>) {
        self.mouse_delta = (pos.x - self.mouse_pos.0, pos.y - self.mouse_pos.1);
        self.mouse_pos = (pos.x, pos.y);

        self.camera.cam_controller.mouse_delta = self.mouse_delta; 
    }





    pub fn update(&mut self) {
        self.camera.update();
        let uniform = self.camera.get_uniform();
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[uniform]));
        if self.camera.cam_controller.clicked {
            let pixel = Vector2::new(self.mouse_pos.0 as f32, self.mouse_pos.1 as f32);
            let uv = pixel.component_div(&Vector2::new(self.config.width as f32, self.config.height as f32));

            let clip_near = Vector4::new(uv.x * 2.0 - 1.0, (uv.y) * 2.0 - 1.0, 0.0, 1.0);
            let clip_far = Vector4::new(uv.x * 2.0 - 1.0, (uv.y) * 2.0 - 1.0, 1.0, 1.0);

            let inv_view_proj = Matrix4::from(uniform.inv_view_proj);

            let mut near = inv_view_proj * clip_near;
            near = near / near.w;      

            let mut far = inv_view_proj * clip_far;
            far = far / far.w;      
            

            let dir = -(far.xyz() - near.xyz()).normalize();

            let cast: Option<u32> = raycast_scene(&self.scene, self.camera.eye, -dir, 500.0, 0.01);
            // cast.inspect(|i| println!("CAST: {i}"));

            self.scene.selected_object = cast.unwrap_or(999);
        
            self.queue.write_buffer(&self.scene_buffer, size_of::<u32>() as u64, bytemuck::cast_slice(&[self.scene.selected_object]));
        }

        self.scene.time = self.start.elapsed().as_secs_f32();
        self.queue.write_buffer(&self.scene_buffer, 2 * size_of::<u32>() as u64, bytemuck::cast_slice(&[self.scene.time]));
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

        let raw_input = self.egui_state.take_egui_input(&self.window);
        let full_output = self.egui_state.egui_ctx().run(raw_input, |ctx| {
            let frame = egui::Frame {
                inner_margin: egui::Margin::same(8),
                corner_radius: egui::CornerRadius::same(4),
                fill: egui::Color32::from_rgb(40, 40, 40),
                shadow: egui::epaint::Shadow::NONE, // <- no shadow
                ..Default::default()
            };

            egui::Window::new("Settings").title_bar(false).resizable(false).anchor(Align2::RIGHT_BOTTOM, egui::vec2(-10.0, -10.0)).frame(frame).show(ctx, |ui| {
                ui.heading(egui::RichText::new("Bold Heading").strong());
                ui.label("Normal text");
            });
        });

        self.egui_state.handle_platform_output(&self.window, full_output.platform_output);
        let tris = self.egui_state.egui_ctx().tessellate(full_output.shapes, self.window.scale_factor() as f32);

        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer.update_texture(
                &self.device,
                &self.queue,
                *id,
                image_delta,
            );
        }

        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &tris,
            &egui_wgpu::ScreenDescriptor {
                size_in_pixels: [self.config.width, self.config.height],
                pixels_per_point: self.window.scale_factor() as f32,
            },
        );

        {
            let mut render_pass = encoder.begin_render_pass(
                &RenderPassDescriptor { 
                    label: Some("Render Pass"), 
                    color_attachments: &[Some(
                        RenderPassColorAttachment { 
                            view: &view, 
                            resolve_target: None, 
                            ops: Operations { 
                                load: LoadOp::Clear(
                                    Color { 
                                        r: 0.0, 
                                        g: 0.0, 
                                        b: 0.0, 
                                        a: 1.0 
                                    }
                                ), 
                                store: StoreOp::Store
                            },
                            depth_slice: None
                        }
                    )], 
                    depth_stencil_attachment: None, 
                    timestamp_writes: None, 
                    occlusion_query_set: None 
                }
            );

            render_pass.set_pipeline(&self.brown_render_pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), IndexFormat::Uint16);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.scene_bind_group, &[]);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }

        { 
            let mut egui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Egui Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None
                })],
                ..Default::default()
            }).forget_lifetime();

            self.egui_renderer.render(
                &mut egui_pass,
                &tris, 
                &egui_wgpu::ScreenDescriptor {
                    size_in_pixels: [self.config.width, self.config.height],
                    pixels_per_point: self.window.scale_factor() as f32,
                },
            );

            for id in &full_output.textures_delta.free {
                self.egui_renderer.free_texture(id);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

