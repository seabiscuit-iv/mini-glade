struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
}

// @builtin(position) is in framebuffer space aka pixel space
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) tex_coords: vec2<f32>
};


struct CameraUniform {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    eye_pos: vec3<f32>
}


struct Ray {
    origin : vec3<f32>,
    direction : vec3<f32>,
}

struct SceneUniform {
    num_objects: u32,
    object_positions: array<vec4<f32>, 512>,
    object_rotations: array<vec4<f32>, 512>,
};

struct SDFResult {
    dist: f32,
    color: vec3<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var<uniform> scene: SceneUniform;

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.color = model.color;

    out.clip_position = vec4<f32>(model.position, 1.0);

    return out;
}


fn apply_euler_rotation(p: vec3<f32>, euler: vec3<f32>) -> vec3<f32> {
    // euler.x = pitch, euler.y = yaw, euler.z = roll
    let cx = cos(euler.x);
    let sx = sin(euler.x);
    let cy = cos(euler.y);
    let sy = sin(euler.y);
    let cz = cos(euler.z);
    let sz = sin(euler.z);

    // rotation matrices for each axis
    let rot_x = mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, cx, -sx),
        vec3<f32>(0.0, sx, cx)
    );

    let rot_y = mat3x3<f32>(
        vec3<f32>(cy, 0.0, sy),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(-sy, 0.0, cy)
    );

    let rot_z = mat3x3<f32>(
        vec3<f32>(cz, -sz, 0.0),
        vec3<f32>(sz, cz, 0.0),
        vec3<f32>(0.0, 0.0, 1.0)
    );

    // apply rotation: Rz * Ry * Rx * p
    return rot_z * (rot_y * (rot_x * p));
}

fn smin(a: SDFResult, b: SDFResult, k: f32) -> SDFResult {
    let h = clamp(0.5 + 0.5 * (b.dist - a.dist) / k, 0.0, 1.0);
    let dist = mix(b.dist, a.dist, h) - k * h * (1.0 - h);
    let color = mix(b.color, a.color, h);
    return SDFResult(dist, color);
}

fn sdf_sphere(p: vec3<f32>, r: f32, color: vec3<f32>) -> SDFResult {
    return SDFResult(length(p) - r, color);
}

fn sdf_box(p: vec3<f32>, b: vec3<f32>, color: vec3<f32>) -> SDFResult {
    let d = abs(p) - b;
    return SDFResult(length(max(d, vec3<f32>(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0), color);
}


const highlighted_obj : u32 = 0u;
const highlight_color : vec3<f32> = vec3(0.9, 0.9, 0.0);

fn scene_sdf(p: vec3<f32>) -> SDFResult {
    var min_dist = SDFResult(5000.0, vec3(0.0));
    var i = 0u;

    while (i < scene.num_objects) {
        var local_p = p - scene.object_positions[i].xyz;

        let r = scene.object_rotations[i];
        local_p = apply_euler_rotation(local_p, scene.object_rotations[i].xyz);

        var sdf = sdf_box(local_p, vec3(1.0), vec3(0.05 * f32(i), 0.0, 1.0 - 0.05 * f32(i)));
        if i == highlighted_obj {
            sdf.color = highlight_color;
        }
        
        min_dist = smin(min_dist, sdf, 0.3);
        i = i + 1u;
    }

    return min_dist;
}

fn calc_normal(p: vec3<f32>) -> vec3<f32> {
    let h = 0.001;
    let k = vec3<f32>(h, 0.0, 0.0);
    let dx = scene_sdf(p + k.xyy).dist - scene_sdf(p - k.xyy).dist;
    let dy = scene_sdf(p + k.yxy).dist - scene_sdf(p - k.yxy).dist;
    let dz = scene_sdf(p + k.yyx).dist - scene_sdf(p - k.yyx).dist;
    return normalize(vec3<f32>(dx, dy, dz));
}

fn get_ray_pos (ray : Ray, t : f32) -> vec3<f32> {
    return ray.origin + t * ray.direction;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.tex_coords * 2.0 - vec2<f32>(1.0, 1.0);
    var clip = vec4<f32>(uv.x, -uv.y, 0.0, 1.0);

    var near = camera.inv_view_proj * clip;
    near /= near.w;

    clip = vec4(uv.x, -uv.y, 1.0, 1.0);
    var far = camera.inv_view_proj * clip;
    far /= far.w;

    let dir = normalize(far.xyz - near.xyz);

    let ray = Ray(
        camera.eye_pos,
        dir
    );

    var t = 0.0;

    var hit = false;
    var dist = 0.0;
 
    var color = vec3(1.0, 0.0, 0.0);

    var hit_pos = vec3<f32>();
    var hit_col = vec3<f32>();

    while t < 500.0 {
        let hit_dist = scene_sdf(get_ray_pos(ray, t));

        if (hit_dist.dist < 0.001) {
            hit = true;
            dist = t;
            hit_pos = get_ray_pos(ray, t);
            hit_col = hit_dist.color;
            break;
        }
        t += hit_dist.dist;
    }

    if !hit {
        return vec4<f32>(0.53, 0.8, 0.92, 1.0);
    }
    else {
        return vec4<f32>(hit_col, 1.0);
    }
}
