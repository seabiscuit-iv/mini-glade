

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
    object_positions: array<vec4<f32>, 1>
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

fn sdf_sphere(p: vec3<f32>, r: f32) -> f32 {
    return length(p) - r;
}

fn scene_sdf(p: vec3<f32>) -> f32 {
    var min_dist = 5000.0;

    var i = 0u;
    while (i < scene.num_objects) {
        min_dist = min(min_dist, sdf_sphere(p + vec3<f32>(f32(i) * 2.0, 0.0, 0.0), 1.0));
        i = i + 1u;
    }

    return min_dist;
}

fn calc_normal(p: vec3<f32>) -> vec3<f32> {
    let h = 0.001;
    let k = vec3<f32>(h, 0.0, 0.0);
    let dx = scene_sdf(p + k.xyy) - scene_sdf(p - k.xyy);
    let dy = scene_sdf(p + k.yxy) - scene_sdf(p - k.yxy);
    let dz = scene_sdf(p + k.yyx) - scene_sdf(p - k.yyx);
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

    while t < 500.0 {
        let hit_dist = scene_sdf(get_ray_pos(ray, t));

        if (hit_dist < 0.0001) {
            hit = true;
            dist = t;
            hit_pos = get_ray_pos(ray, t);
            break;
        }
        t += hit_dist;
    }

    if !hit {
        return vec4<f32>(0.53, 0.8, 0.92, 1.0);
    }
    else {
        return vec4<f32>(normalize(hit_pos), 1.0);
    }
}
