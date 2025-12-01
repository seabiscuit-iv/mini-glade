use std::f32::consts::PI;

use nalgebra::*;

pub struct CameraController {
    pub w: bool,
    pub a: bool,
    pub s: bool,
    pub d: bool,
    pub e: bool,
    pub q: bool,
}


pub struct Camera {
    pub eye: Vector3<f32>,
    pub target: Point3<f32>,
    pub up: Vector3<f32>,
    pub aspect_ratio: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
    pub cam_controller: CameraController
}


impl Camera {
    pub fn from_dimensions(width: u32, height: u32) -> Self {
        Camera {
            eye: [4.0, 4.0, 4.0].into(),
            target: [0.0, 0.0, 0.0].into(),
            up: Vector3::y(),
            aspect_ratio: width as f32 / height as f32,
            fovy: 45.0,
            zfar: 100.0,
            znear: 0.1,
            cam_controller: CameraController {
                w: false,
                a: false,
                d: false,
                e: false,
                q: false,
                s: false
            }
        }
    }

    pub fn build_view_proj_matrix(&self) -> Matrix4<f32>{
        let view = Matrix4::look_at_rh(&self.eye.into(), &self.target, &self.up);

        let persp = Perspective3::new(self.aspect_ratio, self.fovy, self.znear, self.zfar);
        let proj = persp.to_homogeneous();

        let opengl_to_wgpu = Matrix4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.0,
            0.0, 0.0, 0.5, 1.0,
        );

        let view_proj = proj * view;

        opengl_to_wgpu * view_proj
    }

    pub fn update(&mut self) {
        if self.cam_controller.e {
            self.eye.x += 0.01;
        }
        if self.cam_controller.q {
            self.eye.x -= 0.01;
        }
        if self.cam_controller.d {
            self.eye.y += 0.01;
        }
        if self.cam_controller.a {
            self.eye.y -= 0.01;
        }
        if self.cam_controller.w {
            self.eye.z += 0.01
        }
        if self.cam_controller.s {
            self.eye.z -= 0.01
        }
    }

    pub fn get_uniform(&self) -> CameraUniform {
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&self);
        camera_uniform
    }
}



use wgpu::{util::{BufferInitDescriptor, DeviceExt}, *};


#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 3]
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: Matrix4::identity().into(),
            inv_view_proj: Matrix4::identity().into(),
            cam_pos: [1.0; 3]
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_proj_matrix().into();
        self.inv_view_proj = camera.build_view_proj_matrix().try_inverse().unwrap().into();
    }


    pub fn bind_camera(cam: &CameraUniform, device: &Device) -> (Buffer, BindGroupLayout, BindGroup) {
        let camera_buffer = device.create_buffer_init(
            &BufferInitDescriptor { 
                label: Some("Camera Uniform Buffer"), 
                contents: bytemuck::cast_slice(&[*cam]), 
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST 
            }
        );

        let camera_bind_group_layout = device.create_bind_group_layout(
            &BindGroupLayoutDescriptor { 
                label: Some("Camera Bind Group Layout"), 
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::VERTEX,
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

        let camera_bind_group = device.create_bind_group(
            &BindGroupDescriptor { 
                label: Some("Camera Bind Group"), 
                layout: &camera_bind_group_layout, 
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: camera_buffer.as_entire_binding()
                    }
                ]
            }
        );

        (camera_buffer, camera_bind_group_layout, camera_bind_group)
    }
}



fn spherical_to_cartesian(radius: f32, theta: f32, phi: f32) -> Point3<f32> {
    Point3::new(
        radius * phi.sin() * theta.cos(),
        radius * phi.cos(),
        radius * phi.sin() * theta.sin(),
    )
}

