use nalgebra::Vector3;
use wgpu::{util::*, *};

pub const OBJECT_MAX: u32 = 512;


#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneUniform {
    pub num_objects: u32,
    pub padding_0: [f32; 3],
    pub object_positions: [[f32; 4]; OBJECT_MAX as usize]
}


pub fn bind_scene(scene: &SceneUniform, device: &Device) -> (Buffer, BindGroupLayout, BindGroup) 
{
    let scene_buffer = device.create_buffer_init(
        &BufferInitDescriptor {
            label: Some("Scene Buffer"),
            contents: bytemuck::cast_slice(&[*scene]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST 
        }
    );

    let scene_bind_group_layout = device.create_bind_group_layout(
        &BindGroupLayoutDescriptor { 
            label: Some("Scene Bind Group Layout"), 
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    count: None,
                    ty: BindingType::Buffer { 
                        ty: BufferBindingType::Uniform, 
                        has_dynamic_offset: false, 
                        min_binding_size: None 
                    }
                }
            ] 
        }
    );

    let scene_bind_group = device.create_bind_group(
        &BindGroupDescriptor { 
            label: Some("Scene Bind Group"), 
            layout: &scene_bind_group_layout, 
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: scene_buffer.as_entire_binding()
                }
            ]
        }
    );

    (scene_buffer, scene_bind_group_layout, scene_bind_group)
}
