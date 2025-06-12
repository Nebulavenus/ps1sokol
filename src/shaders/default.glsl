#pragma sokol @header import ../math/mat4
#pragma sokol @header import ../math/vec3
#pragma sokol @header import ../math/vec2
#pragma sokol @ctype mat4 Mat4
#pragma sokol @ctype vec3 Vec3
#pragma sokol @ctype vec2 Vec2

#pragma sokol @vs vs
layout(binding = 0) uniform vs_params {
    mat4 u_mvp;
    mat4 u_model;
    vec3 u_camPos;
    float u_jitterAmount;
};

in vec3 a_position;
in vec3 a_normal;
in vec4 a_color0;
in vec2 a_texcoord0;
in float a_ao;

// Sent to the fragment
out vec4 v_color;
// Trick for AFFINE TEXTURE MAPPING
// We'll pass UVs and the clip-space W coordinate together
out vec3 v_affine_uv;
// Distance from vertex to camera for fog
out float v_dist;
out float v_ao;

// Simple directional light
const vec3 lightDir = vec3(0.0, 0.0, -2.0);

void main() {
    // --- 1. Vertex Jitter (GTE Precision Emulation) ---
    // Standard 3D transformation
    vec4 clip_pos = u_mvp * vec4(a_position, 1.0);

    // Snap clip-space coordinates to a lower-resolution grid
    if (u_jitterAmount > 0.0) {
        vec3 ndc = clip_pos.xyz / clip_pos.w; // to Normalized Device Coordinates
        ndc.xy = round(ndc.xy * u_jitterAmount) / u_jitterAmount;
        clip_pos.xyz = ndc * clip_pos.w;
    }
    gl_Position = clip_pos;

    // --- 2. Lighting (Gouraud Shading) ---
    vec3 world_normal = normalize(mat3(u_model) * a_normal);
    float light = 0.3 + max(0.0, dot(world_normal, normalize(lightDir))) * 0.7;
    v_color = vec4(vec3(light), 1.0) * a_color0; // Premultiply it with color from vertex
    // Also pass Ambient Occlusion to fragment
    v_ao = a_ao;

    // --- 3. Fog Calculation ---
    vec3 world_pos = (u_model * vec4(a_position.xyz, 1.0)).xyz;
    v_dist = distance(world_pos, u_camPos);

    // --- 4. The Affine Texture "Wobble" Trick ---
    // To fake affine mapping, we trick the GPU's perspective-correct interpolator
    // 1. Multiply the UVs by the 'w' component of the vertex's clip-space position
    // 2. Pass this new vec3(u*w, v*w, w) to the fragment shader
    //vec2 uv = a_texcoord0 * 5.0;
    vec2 uv = a_texcoord0;
    v_affine_uv = vec3(uv * clip_pos.w, clip_pos.w);
}
#pragma sokol @end

#pragma sokol @fs fs
layout(binding = 0) uniform texture2D u_texture;
layout(binding = 0) uniform sampler u_sampler;

layout(binding = 1) uniform fs_params {
    vec3 u_fogColor;
    float u_fogNear;
    float u_fogFar;
    vec2 u_ditherSize;
    // --- AO Uniforms ---
    float u_aoShadowStrength; // Darkening effect (normal AO) - Multiply
    float u_aoDirtStrength; // Detail color effect - Detail (blood, wet, snow)
    float u_aoTintStrength; // Environment blend effect - Blend (sky ambient blue, cave fungi ambient)
    float u_aoIntensity;
    vec3 u_aoDetailColor;
    vec3 u_aoBaseColor;
};

in vec4 v_color;
in vec3 v_affine_uv;
in float v_dist;
in float v_ao;

out vec4 frag_color;

// --- Dithering & Color Quantization ---
// A classic 4x4 Bayer matrix for ordered dithering
const mat4 bayer_matrix = mat4(
        0.0, 8.0, 2.0, 10.0,
        12.0, 4.0, 14.0, 6.0,
        3.0, 11.0, 1.0, 9.0,
        15.0, 7.0, 13.0, 5.0
    );

// PS1 had 15-bit color (32 levels per channel)
const float colorLevels = 32.0;

vec3 quantize_and_dither(vec3 color, vec2 screen_pos, vec2 dither_size) {
    // 1. Calculate the scale factor between the actual screen size and our virtual dither size.
    // We get the actual screen size from the textureSize of the input uniform (any texture will do).
    // This is more robust than passing in another uniform.
    vec2 actual_size = vec2(textureSize(sampler2D(u_texture, u_sampler), 0));
    vec2 scale_factor = actual_size / dither_size;

    // 2. Scale the screen coordinates. This makes the dither pattern "chunkier" at high resolutions.
    vec2 scaled_coords = floor(screen_pos / scale_factor);

    // 3. Get the dither value from the matrix using the SCALED coordinates.
    float dither_val = bayer_matrix[int(scaled_coords.x) % 4][int(scaled_coords.y) % 4] / 16.0;

    // The rest of the function remains the same...
    float dither_scaled = dither_val / colorLevels;
    vec3 dithered_color = color + dither_scaled;
    return floor(dithered_color * colorLevels) / colorLevels;
}

void main() {
    // --- 1. Affine Texture Sampling, "Wobble" Trick ---
    // Divide the interpolated .xy by the interpolated .z (the original 'w').
    // This undoes the perspective correction for the UVs only, resulting in the
    // classic screen-space linear (affine) interpolation.
    vec2 final_uv = v_affine_uv.xy / v_affine_uv.z;

    // --- 2. Base Color Sampling ---
    // Sample the texture with our wobbly UVs and multiply by the vertex color.
    // The NEAREST filter on the sampler provides the sharp, pixelated look.
    vec4 tex_color = texture(sampler2D(u_texture, u_sampler), final_uv);
    vec3 base_color = tex_color.rgb * v_color.rgb; // Also applies vertex paint
    vec3 final_color = base_color;

    // --- 3. Layered Ambient Occlusion ---

    // Layer 1: Shadows (Multiply)
    // Apply the core darkening effect first
    float shadow_ao = clamp(v_ao * u_aoShadowStrength, 0.0, 1.0);
    final_color *= (1.0 - shadow_ao);

    // Layer 2: Environment Tint (Blend)
    // Lerp between the now-shadowed color and the environment base color
    // Unoccluded areas (where v_ao is low) are tinted the most
    float tint_factor = clamp((1.0 - v_ao) * u_aoTintStrength, 0.0, 1.0);
    final_color = mix(final_color, u_aoBaseColor, tint_factor);

    // Layer 3: Dirt (Detail)
    // Add the detail color on top of everything else
    float dirt_ao = clamp(v_ao * u_aoDirtStrength, 0.0, 1.0);
    final_color += dirt_ao * u_aoDetailColor;

    // --- 4. Color Quantization and Dithering ---
    final_color = quantize_and_dither(final_color, gl_FragCoord.xy, u_ditherSize.xy);

    // --- 5. Fog Application ---
    float fog_factor = smoothstep(u_fogNear, u_fogFar, v_dist);
    final_color = mix(final_color, u_fogColor.rgb, fog_factor);

    // Final output
    frag_color = vec4(final_color, 1.0);

    // Debug
    //frag_color = vec4(final_uv.x, final_uv.y, 0.0, 1.0);
}
#pragma sokol @end

#pragma sokol @program texcube vs fs
