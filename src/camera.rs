use std::f32::consts::PI;

use nalgebra::*;

pub struct CameraController {
    pub w: bool,
    pub a: bool,
    pub s: bool,
    pub d: bool,
    pub e: bool,
    pub q: bool,
    pub dragging: bool,
    pub mouse_delta : (f64, f64)
}


pub struct Camera {
    pub eye: Vector3<f32>,
    pub look: Vector3<f32>,
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
            look: [0.0, 0.0, 1.0].into(),
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
                s: false,
                dragging: false,
                mouse_delta: (0.0, 0.0)
            }
        }
    }

    pub fn build_view_proj_matrix(&self) -> Matrix4<f32>{
        let target = self.look + self.eye;
        let view = Matrix4::look_at_rh(&self.eye.into(), &target.into(), &self.up);

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
        let right = Vector3::normalize(&self.look.cross(&self.up));

        if self.cam_controller.e {
            self.eye += self.up * 0.01;
        }
        if self.cam_controller.q {
            self.eye -= self.up * 0.01;
        }
        if self.cam_controller.d {
            self.eye += -right * 0.01;
        }
        if self.cam_controller.a {
            self.eye -= -right * 0.01;
        }
        if self.cam_controller.w {
            self.eye += -self.look * 0.01;
        }
        if self.cam_controller.s {
            self.eye -= -self.look * 0.01;
        }
    
        if self.cam_controller.dragging {
            let sensitivity = 0.002_f32; // tweak this
            let dx = self.cam_controller.mouse_delta.0 as f32;
            let dy = self.cam_controller.mouse_delta.1 as f32;

            // yaw (around world Y), pitch (around camera-local X)
            let yaw = -dx * sensitivity;
            let pitch = -dy * sensitivity;

            // build a quaternion: first pitch (x), then yaw (y)
            let q = UnitQuaternion::from_euler_angles(pitch, yaw, 0.0);

            // rotate the look vector
            self.look = (q * self.look).normalize();
        }

        self.cam_controller.mouse_delta = (0.0, 0.0);
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
    cam_pos: [f32; 4]
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: Matrix4::identity().into(),
            inv_view_proj: Matrix4::identity().into(),
            cam_pos: [1.0; 4]
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera) {
        let vp = camera.build_view_proj_matrix();
        self.view_proj = vp.into();
        self.inv_view_proj = vp.try_inverse().unwrap().into();

        self.cam_pos = [camera.eye.x, camera.eye.y, camera.eye.z, 0.0];
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

