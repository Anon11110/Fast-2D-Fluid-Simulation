#version 450

layout(location=0) in vec2 screen_uv;
layout(location=0) out vec4 frag_color;

layout(set=0, binding=0) uniform sampler2D velocity_texture;
layout(set=0, binding=1) uniform sampler2D divergence_texture;
layout(set=0, binding=2) uniform sampler2D pressure_texture;
layout(set=0, binding=3) uniform sampler2D color_texture;

vec3 ACES(vec3 color) 
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;

    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
}

void main() 
{
    vec2 velocity = texture(velocity_texture, screen_uv).rg;
    float pressure = texture(pressure_texture, screen_uv).r;
    vec4 color = texture(color_texture, screen_uv);

    color.rgb = ACES(color.rgb);

    frag_color = color;
}