#version 450

layout(location=0) in vec2 screen_uv;
layout(location=0) out vec4 frag_color;

layout(set=0, binding=0) uniform sampler2D color_texture;
layout(set=0, binding=1) uniform sampler2D obstacle_mask_texture;

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
    vec4 fluid_color = texture(color_texture, screen_uv);
    float obstacle = texture(obstacle_mask_texture, screen_uv).r;

    vec4 obstacle_color = vec4(1.0);

    fluid_color.rgb = ACES(fluid_color.rgb);

    if (obstacle > 0.5)
    {
        frag_color = obstacle_color;
    }
    else
    {
        frag_color = fluid_color;
    }
}