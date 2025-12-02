use std::sync::Arc;
use iced_winit::winit::window::Window;
use iced_wgpu::wgpu;
use iced_wgpu::wgpu::*;
use crate::render;
use crate::shader_structs::Vertex;

//helper fn for render pipeline descriptors
pub fn make_pipeline_desc_from_shader(device: &Device, layout: &PipelineLayout, shader: &ShaderModule, fmt: TextureFormat) -> RenderPipeline {
    let vertex_buffer_layout = Vertex::desc();
    
    device.create_render_pipeline(
        &RenderPipelineDescriptor { 
            label: Some("Render Pipeline"), 
            layout: Some(layout), 
            vertex: VertexState { 
                module: shader, 
                entry_point: "vs_main", 
                buffers: &[vertex_buffer_layout] 
            }, 
            fragment: Some(FragmentState {
                module: shader,
                entry_point: "fs_main",
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
        instance_descriptor
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
    .await.unwrap();

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
        },
        None
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