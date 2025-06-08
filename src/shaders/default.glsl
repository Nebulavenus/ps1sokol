#pragma sokol @header import ../math/mat4
#pragma sokol @ctype mat4 Mat4

#pragma sokol @vs vs
layout(binding = 0) uniform vs_params {
    mat4 mvp;
};

in vec4 position;
in vec4 color0;
in vec2 texcoord0;

// Sent to the fragment
out vec4 color;
// Trick for AFFINE TEXTURE MAPPING
// We'll pass UVs and the clip-space W coordinate together
out vec3 affine_uv;

void main() {
    // Standard 3D transformation
    vec4 clip_pos = mvp * position;
    gl_Position = clip_pos;

    // Pass vertex color for Gouraud shading
    color = color0;

    // --- The Affine Texture "Wobble" Trick ---
    // To fake affine mapping, we trick the GPU's perspective-correct interpolator
    // 1. Multiply the UVs by the 'w' component of the vertex's clip-space position
    // 2. Pass this new vec3(u*w, v*w, w) to the fragment shader
    vec2 uv = texcoord0 * 2.0;
    affine_uv = vec3(uv * clip_pos.w, clip_pos.w);
}
#pragma sokol @end

#pragma sokol @fs fs
layout(binding = 0) uniform texture2D tex;
layout(binding = 0) uniform sampler smp;

in vec4 color;
in vec3 affine_uv;

out vec4 frag_color;

void main() {
    // --- The Affine Texture "Wobble" Trick (Part 2) ---
    // 3. Divide the interpolated .xy by the interpolated .z (the original 'w').
    // This undoes the perspective correction for the UVs only, resulting in the
    // classic screen-space linear (affine) interpolation.
    vec2 final_uv = affine_uv.xy / affine_uv.z;

    // Sample the texture with our wobbly UVs and multiply by the vertex color.
    // The NEAREST filter on the sampler provides the sharp, pixelated look.
    frag_color = texture(sampler2D(tex, smp), final_uv) * color;
}
#pragma sokol @end

#pragma sokol @program texcube vs fs
