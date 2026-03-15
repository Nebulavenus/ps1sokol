#pragma sokol @vs vs
in vec2 position;
in vec2 texcoord0;
out vec2 uv;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    uv = texcoord0;
}
#pragma sokol @end

#pragma sokol @fs fs
layout(binding = 0) uniform texture2D u_texture;
layout(binding = 0) uniform sampler u_sampler;

layout(binding = 0) uniform fs_params {
    vec2 u_resolution;
    float u_time;
};

in vec2 uv;
out vec4 frag_color;

void main() {
    vec4 tex_color = texture(sampler2D(u_texture, u_sampler), uv);
    
    // Simple CRT scanlines
    float scanline = sin(uv.y * u_resolution.y * 1.5) * 0.1 + 0.9;
    
    // Slight color grading (PS1-ish)
    vec3 color = tex_color.rgb;
    color *= vec3(1.05, 1.0, 0.95); // Warm tint
    color = floor(color * 16.0) / 16.0; // Color depth reduction
    
    frag_color = vec4(color * scanline, 1.0);
}
#pragma sokol @end

#pragma sokol @program postfx vs fs
