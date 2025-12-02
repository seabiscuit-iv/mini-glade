use std::sync::Arc;
use winit::window::Window;
use wgpu::*;
use crate::shader_structs::Vertex;


pub fn with_default_render_pass<F>(
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    draw_fn: F,
) 
where
    F: FnOnce(&mut wgpu::RenderPass),
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
                    depth_slice: None, 
                }
            )], 
            depth_stencil_attachment: None, 
            timestamp_writes: None, 
            occlusion_query_set: None 
        }
    );

    draw_fn(&mut render_pass);
}

//helper fn for render pipeline descriptors
pub fn make_pipeline_desc_from_shader(device: &Device, layout: &PipelineLayout, shader: &ShaderModule, fmt: TextureFormat) -> RenderPipeline {
    let vertex_buffer_layout = Vertex::desc();
    
    device.create_render_pipeline(
        &RenderPipelineDescriptor { 
            label: Some("Render Pipeline"), 
            layout: Some(layout), 
            vertex: VertexState { 
                module: shader, 
                entry_point: Some("vs_main"), 
                compilation_options: PipelineCompilationOptions::default(), 
                buffers: &[vertex_buffer_layout] 
            }, 
            fragment: Some(FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(), 
                targets: &[Some(ColorTargetState { 
                    format: fmt, 
                    blend: Some(BlendState::REPLACE), 
                    write_mask: ColorWrites::ALL 
                })]
            }),
            primitive: PrimitiveState { 
                topology: PrimitiveTopology::TriangleList, 
                strip_index_format: None, 
                front_face: FrontFace::Ccw, 
                cull_mode: Some(Face::Back), 
                unclipped_depth: false, 
                polygon_mode: PolygonMode::Fill, 
                conservative: false 
            }, 
            depth_stencil: None,
            multiview: None, 
            cache: None,
            multisample: MultisampleState { 
                count: 1, 
                mask: !0, 
                alpha_to_coverage_enabled: false 
            }, 
        }
    )
}

pub async fn configure_surface(window: Arc<Window>) -> anyhow::Result<(Surface<'static>, SurfaceConfiguration, Device, Queue)> {
    let size = window.inner_size();

    let instance_descriptor = 
        InstanceDescriptor {
            #[cfg(target_arch = "wasm32")]
            backends: Backends::BROWSER_WEBGPU,
            #[cfg(not(target_arch = "wasm32"))]
            backends: Backends::PRIMARY,
            ..Default::default()
        };

    let instance = Instance::new(
        &instance_descriptor
    );

    let window_ref = window.clone();
    let surface = instance.create_surface(window_ref).unwrap();

    // adapter is a handle for our graphics card
    let adapter = instance.request_adapter( 
        &RequestAdapterOptions { 
            power_preference: PowerPreference::HighPerformance, 
            force_fallback_adapter: false, 
            compatible_surface: Some(&surface)
        }
    )
    .await?;

    println!("GPU: {}", adapter.get_info().name);

    let (device, queue) = adapter.request_device(
        &DeviceDescriptor { 
            label: None, 
            required_features: Features::empty(), 
            required_limits: 
                if cfg!(target_arch = "wasm32") {
                    Limits::downlevel_defaults()
                } else {
                    Limits::default()
                }
            , 
            memory_hints: Default::default(), 
            trace: Trace::Off 
        }
    )
    .await?;

    let surface_caps = surface.get_capabilities(&adapter);

    let surface_fmt = surface_caps.formats.iter()
        .find(|f| f.is_srgb())
        .copied()
        .unwrap_or(surface_caps.formats[0]);

    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_fmt,
        
        #[cfg(not(target_arch = "wasm32"))]
        width: size.width,
        #[cfg(not(target_arch = "wasm32"))]
        height: size.height,

        #[cfg(target_arch = "wasm32")]
        width: size.width,
        #[cfg(target_arch = "wasm32")]
        height: size.height,

        present_mode: surface_caps.present_modes[0],
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };

    Ok((surface, config, device, queue))
}