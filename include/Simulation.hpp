#pragma once

#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include "imgui.h"
#include "liblava/lava.hpp"

namespace FluidSimulation
{
struct FrameTimeInfo
{
    double current_time{};
    float delta_time{};
};

struct SimulationConstants
{
    float current_time;
    float delta_time;
    int texture_width;
    int texture_height;
    float fluid_density;
    float vorticity_strength;
    bool reset_color;
};

class Simulation
{
  public:
    Simulation(lava::engine &app);

    ~Simulation();

    void OnUpdate(VkCommandBuffer cmd_buffer, const FrameTimeInfo &frame_context);

    [[nodiscard]] lava::texture::s_ptr GetVelocityTexture() const
    {
        return velocity_field_texture_;
    }

    bool reset_flag_ = true;

    using s_ptr = std::shared_ptr<Simulation>;

    static s_ptr make(lava::engine &app)
    {
        return std::make_shared<Simulation>(app);
    }

    friend class FluidRenderer;

  private:
    void AddShaderMappings();
    void CreateTextures();
    void CreateDescriptorPool();
    void CreateDescriptorSets();
    void SetupPipelines();
    void UpdateDescriptorSets();

    lava::engine &app_;

    lava::descriptor::pool::s_ptr descriptor_pool_;

    // Velocity advection
    lava::texture::s_ptr velocity_field_texture_;
    lava::texture::s_ptr advected_velocity_field_texture_;
    lava::descriptor::s_ptr advect_descriptor_set_layout_;
    VkDescriptorSet advect_descriptor_set_{};
    lava::pipeline_layout::s_ptr advect_pipeline_layout_;
    lava::compute_pipeline::s_ptr advect_pipeline_;


};
}

#endif // SIMULATION_HPP