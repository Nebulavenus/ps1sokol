#pragma sokol @header import ../math/mat4
#pragma sokol @ctype mat4 Mat4

#pragma sokol @vs vs
layout(binding = 0) uniform vs_params {
    mat4 mvp;
};

in vec4 position;
in vec4 color0;
in vec2 texcoord0;

out vec4 color;
out vec2 uv;

void main() {
    gl_Position = mvp * position;
    color = color0;
    uv = texcoord0 * 5.0;
}
#pragma sokol @end

#pragma sokol @fs fs
layout(binding = 0) uniform texture2D tex;
layout(binding = 0) uniform sampler smp;

in vec4 color;
in vec2 uv;
out vec4 frag_color;

void main() {
    frag_color = texture(sampler2D(tex, smp), uv) * color;
}
#pragma sokol @end

#pragma sokol @program texcube vs fs
