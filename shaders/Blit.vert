#version 450

layout(location=0) out vec2 screen_uv;

void main() 
{
    vec2 screen_pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    screen_uv = screen_pos;
    gl_Position = vec4(screen_pos * 2.0 - 1.0, 0.0, 1.0);
}
