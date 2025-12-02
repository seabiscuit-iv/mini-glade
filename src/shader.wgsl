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
    selected_object: u32,
    time: f32,
    object_positions: array<vec4<f32>, 64>,
    object_rotations: array<vec4<f32>, 64>,
    object_meta: array<vec4<u32>, 64>,
    object_param_1: array<vec4<f32>, 64>,
    object_param_2: array<vec4<f32>, 64>,
    object_param_3: array<vec4<f32>, 64>,
    object_param_4: array<vec4<f32>, 64>,
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


fn from_rgb(r: i32, g: i32, b: i32) -> vec3<f32> {
    return vec3<f32>(f32(r), f32(g), f32(b)) / 255.0f;
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

fn sdf_min(a: SDFResult, b: SDFResult) -> SDFResult {
    if a.dist < b.dist {
        return a;
    }
    else {
        return b;
    }
}

fn sdf_sphere(p: vec3<f32>, r: f32, color: vec3<f32>) -> SDFResult {
    return SDFResult(length(p) - r, color);
}

fn sdf_box(p: vec3<f32>, b: vec3<f32>, color: vec3<f32>) -> SDFResult {
    let d = abs(p) - b;
    return SDFResult(length(max(d, vec3<f32>(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0), color);
}





fn hash22(p : vec2<f32>) -> vec2<f32> {
    let x = sin(p.x * 127.1 + p.y * 311.7) * 43758.5453;
    let y = sin(p.x * 269.5 + p.y * 183.3) * 43758.5453;
    return fract(vec2(x, y));
}

fn hash(n: vec2<f32>) -> f32 {
    return fract(sin(dot(n, vec2(127.1, 311.7))) * 43758.5453123);
}

fn voronoi(uv: vec2<f32>) -> vec2<f32>
{
    let f = fract(uv);
    let u = floor(uv);
    
    var closest = 100.0;
    var id = 0.0;
    for (var y = -1; y <= 1; y += 1)
    {
        for (var x = -1; x <= 1; x += 1)
        {
            var d = vec2(f32(x), f32(y));
            var nu = u + d;
            var p = hash22(nu);
            let dist = distance(f, p + d);
            if (dist < closest)
            {
                closest = dist;
                id = hash(nu);
            }
        }
    }
    return vec2(closest, id);
}

fn voronoi_3d(p: vec3<f32>) -> f32
{
    let grassHeight = 1.0;
    
    // 1. Scale p.xz for the 2D Voronoi grid
    let uv = p.xz * 3.0;

    // Get (Closest_Distance_D, ID)
    let vor = voronoi(uv); // Using the hypothetically modified function
    let closest_distance_D = vor.x;
    let id = vor.y;

    // 2. Determine the grass blade's "thickness" or threshold at this height (p.y)
    // The surface is defined where the cell boundary shrinks to the cell center.
    // The original boundary value scales from 0 (at p.y=0) to 1 (at p.y=grassHeight).
    let boundary_threshold = 1.0 - max(p.y / grassHeight, 0.0);
    
    // 3. Calculate the Signed Distance
    // The SDF is the difference between the closest distance (D) and the threshold.
    // D < threshold (Inside the blade) -> Negative SDF
    // D = threshold (On the surface) -> Zero SDF
    // D > threshold (Outside the blade) -> Positive SDF
    let sdf = closest_distance_D - boundary_threshold;

    // 4. Distance Scaling (Crucial for a "true" SDF)
    // The D value is in the range [0, sqrt(2)/2] (distance across a unit cell).
    // The sdf value needs to be scaled to approximate the Euclidean distance.
    // Since the max size of the SDF is related to 1 / scale_factor, we scale up.
    let scale_factor = 3.0; // Same as the p.xz * 3.0 scaling
    return sdf / scale_factor;
}


fn voronoiGrassSDF(p: vec3<f32>) -> SDFResult {
    let grassHeight = 1.0;
    
    // --- 1. SDF Calculation (from your original logic) ---
    let uv = p.xz * 3.0;
    let vor = voronoi(uv);
    let closest_distance_D = vor.x;
    let cell_id = vor.y; // <--- This is the stable per-blade ID

    let boundary_threshold = 1.0 - max(p.y / grassHeight, 0.0);
    let sdf = closest_distance_D - boundary_threshold;
    let scale_factor = 3.0;
    let dist = sdf / scale_factor;

    // --- 2. Color Calculation ---

    // Define color ranges
    let id_factor = clamp(cell_id * 2.0 - 1.0, 0.0, 1.0);

    let dark_green = from_rgb(30, 100 + i32(30 * id_factor) , 30);  // Darker, near the root
    let bright_green = from_rgb(134, 212 + i32(12 * (1.0 - id_factor)), 148); // Lighter, near the tip

    let height_factor = clamp(p.y / grassHeight, 0.0, 1.0);
    var color = mix(dark_green, bright_green, height_factor);
    
    return SDFResult(dist, color); // Return the calculated color
}


const highlight_color : vec3<f32> = vec3(0.9, 0.9, 0.0);

fn basic_scene_sdf(p: vec3<f32>) -> SDFResult {
    let ground = SDFResult(p.y, from_rgb(124, 182, 142));;
    let planter = sdf_box(p - vec3(0.0, 1.0, 0.0), vec3(5.0, 1.0, 3.0), from_rgb(51, 36, 33));
    let soil = sdf_box(p - vec3(0.0, 1.1, 0.0), vec3(4.5, 1.0, 2.6), from_rgb(38, 28, 15));
    return sdf_min(sdf_min(sdf_min(ground, planter), soil), voronoiGrassSDF(p));
}


fn pick_sdf(p: vec3<f32>, metadata: vec4<u32>, param1: vec4<f32>, param2: vec4<f32>, param3: vec4<f32>, param4: vec4<f32>) -> SDFResult {
    let obj_type = metadata.x;

    if obj_type == 0 {
        return sdf_box(p, param1.xyz, param2.xyz); // size, color
    }
    else if obj_type == 1 {
        return sdf_sphere(p, param1.x, param2.xyz);
    }
    else {
        return sdf_sphere(p, 1.0, from_rgb(0, 0, 0));
    }
}


fn scene_sdf(p: vec3<f32>) -> SDFResult {
    var min_dist = SDFResult(5000.0, vec3(0.0));
    var i = 0u; 

    while (i < scene.num_objects) {
        var local_p = p - scene.object_positions[i].xyz;

        let r = scene.object_rotations[i];
        local_p = apply_euler_rotation(local_p, scene.object_rotations[i].xyz);



        var sdf = sdf_box(local_p, vec3(1.0), vec3(0.05 * f32(i), 0.0, 1.0 - 0.05 * f32(i)));



        if i == scene.selected_object {
            sdf.color = highlight_color;
        }
        
        min_dist = smin(min_dist, sdf, 0.3);
        i = i + 1u;
    }

    return sdf_min(min_dist, basic_scene_sdf(p));
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








fn ambient_occlusion(p: vec3<f32>, n: vec3<f32>) -> f32 {
    var occlusion = 0.0;
    let scale = 0.4;
    for (var i = 1; i <= 5; i += 1) {
        let t = 0.045 * f32(i); // step along normal
        let d = scene_sdf(p + n * t); // SDF distance
        occlusion += (t - d.dist);      // closer surfaces contribute more
    }
    return clamp(1.0 - occlusion, 0.0, 1.0);
}







fn rgb2hsv(c: vec3<f32>) -> vec3<f32> {
    let rgb_min = min(min(c.r, c.g), c.b);
    let rgb_max = max(max(c.r, c.g), c.b);
    let delta = rgb_max - rgb_min;

    var h = 0.0;
    if (delta != 0.0) {
        if (c.r == rgb_max) {
            h = (c.g - c.b) / delta;
        } else if (c.g == rgb_max) {
            h = 2.0 + (c.b - c.r) / delta;
        } else {
            h = 4.0 + (c.r - c.g) / delta;
        }
        h = h * 60.0;
    }
    if (h < 0.0) {
        h = h + 360.0;
    }
    
    let s = select(0.0, delta / rgb_max, rgb_max != 0.0);
    let v = rgb_max;
    
    return vec3(h / 360.0, s, v);
}

fn hsv2rgb(c: vec3<f32>) -> vec3<f32> {
    let k = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = abs(fract(c.xxx + k.xyz) * 6.0 - k.www);
    let rgb = c.z * mix(k.xxx, clamp(p - k.x, vec3(0.0), vec3(1.0)), c.y);
    return rgb;
}

// -- Simplified Reinhard Tonemapping Function --
// Compresses high values while maintaining midtone contrast.
fn reinhard(color: vec3<f32>) -> vec3<f32> {
    // Applied component-wise
    return color / (vec3(1.0) + color);
}

// -- Saturation Boosting Function --
// Increases the S channel in HSV space.
fn boost_saturation(color: vec3<f32>, factor: f32) -> vec3<f32> {
    let hsv = rgb2hsv(color);
    var boosted_hsv = hsv;
    
    // Multiply the saturation (y component)
    boosted_hsv.y = clamp(hsv.y * factor, 0.0, 1.0);
    
    return hsv2rgb(boosted_hsv);
}


fn ACESFilm( x: vec3<f32> ) -> vec3<f32>
{
    let tA = 2.51;
    let tB = 0.03;
    let tC = 2.43;
    let tD = 0.59;
    let tE = 0.14;
    return clamp((x*(tA*x+tB))/(x*(tC*x+tD)+tE),vec3(0.0),vec3(1.0));
}

const RayleighAtt : f32 = 1.0;  
const MieAtt : f32 = 1.2;

fn sky(_o: vec3<f32>, _d: vec3<f32>, sun_dir: vec3<f32> ) -> vec3<f32> {
    var origin = _o;
    var dir = _d;
    var color = vec3(0.0);

    let day_night = clamp(sun_dir.y * 6.0, 0.1, 1.0);
    
	if dir.y < 0 {
		let L = - origin.y / dir.y;
		origin = origin + dir * L;
        dir.y = -dir.y;
		dir = normalize(dir);
	}
    else{
     	let L1 =  origin.y / origin.y;
		let O1 = origin + dir * L1;

    	var D1 = vec3(1.0);
    	D1 = normalize(dir);
    }
    
    let t = max(0.001, dir.y) + max(-dir.y, -0.001);

    // optical depth -> zenithAngle
    let sR = RayleighAtt / t ;
    let sM = MieAtt / t ;

    let cosine = clamp(dot(dir, sun_dir),0.0,1.0);
    let extinction = exp(-(vec3(1.95e-2, 1.1e-1, 2.94e-1) * sR + vec3(4e-2, 4e-2, 4e-2) * sM));

    // scattering phase
    let g2 = -0.9 * -0.9;
    let fcos2 = cosine * cosine;
    let miePhase = 1.0 * pow(1. + g2 + 2. * -0.9 * cosine, -1.5) * (1. - g2) / (2. + g2);

    let rayleighPhase = 1.0;

    let inScatter = (1. + fcos2) * vec3(rayleighPhase + vec3(4e-2, 4e-2, 4e-2) / vec3(1.95e-2, 1.1e-1, 2.94e-1) * miePhase);

    color = inScatter*(1.0-extinction); // *vec3(1.6,1.4,1.0)

    // sun
    color += 0.47*vec3(1.6,1.4,1.0)*pow( cosine, 350.0 ) * extinction;
    // sun haze
    color += 0.4*vec3(0.8,0.9,1.0)*pow( cosine, 2.0 )* extinction;

    color *= day_night;
    
    color = ACESFilm(color);

    color = pow(color, vec3(2.2));

    return color;
}



@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.tex_coords * 2.0 - vec2<f32>(1.0, 1.0);
    var clip = vec4<f32>(uv.x, -uv.y, 0.0, 1.0);

    let time = scene.time * 0.02; // slow it down

    let light_pos = normalize(vec3(
        sin(time),
        cos(time),
        0.0
    ));

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
        return vec4<f32>(sky(ray.origin, ray.direction, light_pos), 1.0);
    }
    else {
        let normal = calc_normal(hit_pos);

        let lambert = max(dot(light_pos, normal), 0.0);

        let ao = clamp(1.0 - ambient_occlusion(hit_pos, normal), 0.0, 0.3);

        let light_contrib = (lambert - ao) * clamp(sky(ray.origin, ray.direction, light_pos), vec3(0.0), vec3(1.0));
        // let light_contrib = lambert;

        let col = vec3<f32>(hit_col * (light_contrib * 0.7 + 0.3));

        let tonemapped_rgb = reinhard(col);
        let saturation_factor = 1.2;
        let final_rgb = boost_saturation(tonemapped_rgb, saturation_factor);
        let final_col = vec4<f32>(final_rgb, 1.0);

        return final_col;
    }
}
