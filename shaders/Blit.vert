#version 450

layout(location=0) out vec2 screen_uv;

void main() 
{
    const vec2 positions[3] = vec2[](
        vec2(-1.0, -1.0),
        vec2(3.0, -1.0),
        vec2(-1.0, 3.0)
    );

    screen_uv = 0.5 * positions[gl_VertexIndex] + 0.5;
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
