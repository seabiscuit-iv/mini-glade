use nalgebra::{Matrix3, Vector3};
use wgpu::{util::*, *};

pub const OBJECT_MAX: u32 = 512;


#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneUniform {
    pub num_objects: u32,
    pub selected_object: u32,
    pub padding_0: [f32; 2],
    pub object_positions: [[f32; 4]; OBJECT_MAX as usize],
    pub object_rotations: [[f32; 4]; OBJECT_MAX as usize],
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

pub fn raycast_scene(
    scene: &SceneUniform,
    ray_origin: Vector3<f32>,
    ray_dir: Vector3<f32>,
    max_dist: f32,
    epsilon: f32,
) -> Option<u32> {
    let mut t = 0.0;

    while t < max_dist {
        let mut closest_dist = f32::MAX;
        let mut hit_id = None;

        for i in 0..scene.num_objects as usize {
            // Get object position
            let pos = Vector3::new(
                scene.object_positions[i][0],
                scene.object_positions[i][1],
                scene.object_positions[i][2],
            );

            // Transform ray into object-local space
            let local_pos = ray_origin + ray_dir * t - pos;

            // Get object rotation (Euler angles in radians)
            let euler = Vector3::new(
                scene.object_rotations[i][0],
                scene.object_rotations[i][1],
                scene.object_rotations[i][2],
            );

            // Build rotation matrices
            let (cx, sx) = (euler.x.cos(), euler.x.sin());
            let (cy, sy) = (euler.y.cos(), euler.y.sin());
            let (cz, sz) = (euler.z.cos(), euler.z.sin());

            let rot_x = Matrix3::new(
                1.0, 0.0, 0.0,
                0.0, cx, -sx,
                0.0, sx, cx,
            );

            let rot_y = Matrix3::new(
                cy, 0.0, sy,
                0.0, 1.0, 0.0,
                -sy, 0.0, cy,
            );

            let rot_z = Matrix3::new(
                cz, -sz, 0.0,
                sz, cz, 0.0,
                0.0, 0.0, 1.0,
            );

            // Full rotation matrix: Rz * Ry * Rx
            let rotation = rot_z * rot_y * rot_x;

            // Apply inverse rotation to local position
            let local_pos = rotation.transpose() * local_pos;

            // Unit box SDF
            let q = local_pos.abs() - Vector3::new(1.0, 1.0, 1.0);
            let box_dist = q.map(|v| v.max(0.0)).norm() + q.x.max(q.y.max(q.z)).min(0.0);

            if box_dist < closest_dist {
                closest_dist = box_dist;
                hit_id = Some(i as u32);
            }
        }

        t += closest_dist.max(epsilon); // avoid infinite loop

        if closest_dist < epsilon {
            return hit_id;
        }
    }

    None
}
