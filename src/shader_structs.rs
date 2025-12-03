use bytemuck::{Pod, Zeroable};
use egui_wgpu::wgpu;
use wgpu::*;
use nalgebra::*;


#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
    tex_coords: [f32; 2]
}
pub const VERTICES: &[Vertex] = &[
    Vertex { position: [-1.0, -1.0, 0.0], color: [1.0, 1.0, 1.0], tex_coords: [0.0, 0.0] }, // bottom-left
    Vertex { position: [ 1.0, -1.0, 0.0], color: [1.0, 1.0, 1.0], tex_coords: [1.0, 0.0] }, // bottom-right
    Vertex { position: [ 1.0,  1.0, 0.0], color: [1.0, 1.0, 1.0], tex_coords: [1.0, 1.0] }, // top-right
    Vertex { position: [-1.0,  1.0, 0.0], color: [1.0, 1.0, 1.0], tex_coords: [0.0, 1.0] }, // top-left
];

pub const INDICES: &[u16] = &[
    0, 1, 2,  // first triangle
    2, 3, 0,  // second triangle
];

impl Vertex {
    const ATTRIBS : [VertexAttribute; 3] = vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x2
    ];

    pub fn desc() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS
        }
    }
}


