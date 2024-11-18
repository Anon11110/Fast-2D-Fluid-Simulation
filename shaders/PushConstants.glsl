layout(push_constant) uniform SimulationPushConstants
{
    float current_time;
    float delta_time;
    int texture_width;
    int texture_height;
    float fluid_density;
    float vorticity_strength;
    bool reset_flag;
} push_constants;