#pragma sokol @header import ../math/mat4
#pragma sokol @header import ../math/vec3
#pragma sokol @header import ../math/vec2
#pragma sokol @ctype mat4 Mat4
#pragma sokol @ctype vec3 Vec3
#pragma sokol @ctype vec2 Vec2

#pragma sokol @vs vs
layout(binding = 0) uniform vs_params {
    mat4 u_mvp;
    vec3 u_camPos;
    float u_jitterAmount;
};

in vec3 a_position;
in vec4 a_color0;
in vec2 a_texcoord0;

out vec4 v_color;
out vec3 v_affine_uv;
out float v_dist;

void main() {
    vec4 clip_pos = u_mvp * vec4(a_position, 1.0);

    if (u_jitterAmount > 0.0) {
        vec3 ndc = clip_pos.xyz / clip_pos.w;
        ndc.xy = round(ndc.xy * u_jitterAmount) / u_jitterAmount;
        clip_pos.xyz = ndc * clip_pos.w;
    }
    gl_Position = clip_pos;

    v_color = a_color0;
    v_dist = distance(a_position, u_camPos);
    v_affine_uv = vec3(a_texcoord0 * clip_pos.w, clip_pos.w);
}
#pragma sokol @end

#pragma sokol @fs fs
layout(binding = 0) uniform texture2D u_texture;
layout(binding = 0) uniform sampler u_sampler;

layout(binding = 1) uniform fs_params {
    vec3 u_fogColor;
    float u_fogNear;
    float u_fogFar;
    float u_alphaThreshold;
};

in vec4 v_color;
in vec3 v_affine_uv;
in float v_dist;

out vec4 frag_color;

void main() {
    vec2 final_uv = v_affine_uv.xy / v_affine_uv.z;
    vec4 tex_color = texture(sampler2D(u_texture, u_sampler), final_uv);
    
    if (tex_color.a < u_alphaThreshold) {
        discard;
    }

    vec3 final_color = tex_color.rgb * v_color.rgb;
    float fog_factor = smoothstep(u_fogNear, u_fogFar, v_dist);
    final_color = mix(final_color, u_fogColor.rgb, fog_factor);

    frag_color = vec4(final_color, tex_color.a * v_color.a);
}
#pragma sokol @end

#pragma sokol @program sprite vs fs
