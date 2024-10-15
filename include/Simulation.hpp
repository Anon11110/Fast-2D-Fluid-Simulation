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

enum class PressureProjectionMethod
{
    Jacobi,
    Kernel
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

    [[nodiscard]] lava::texture::s_ptr GetDivergenceTexture() const
    {
        return divergence_field_texture_;
    }

    [[nodiscard]] lava::texture::s_ptr GetPressureTexture() const
    {
        return pressure_field_jacobi_texture_A_;
    }

    [[nodiscard]] lava::texture::s_ptr GetColorTexture() const
    {
        return color_field_texture_A_;
    }

    [[nodiscard]] PressureProjectionMethod GetPressureProjectionMethod() const
    {
        return pressure_projection_method_;
    }

    void SetPressureProjectionMethod(const PressureProjectionMethod &method)
    {
        pressure_projection_method_ = method;
    }

    [[nodiscard]] uint32_t GetPressureJacobiIterations() const
    {
        return pressure_jacobi_iterations_;
    }

    void SetPressureJacobiIterations(uint32_t iterations)
    {
        pressure_jacobi_iterations_ = iterations;
    }

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
    void JacobiPressureProjection(VkCommandBuffer cmd_buffer, const SimulationConstants &constants);
    void KernelPressureProjection(VkCommandBuffer cmd_buffer, const SimulationConstants &constants);

    lava::engine &app_;

    lava::descriptor::pool::s_ptr descriptor_pool_;

    bool reset_flag_ = true;

    PressureProjectionMethod pressure_projection_method_ = PressureProjectionMethod::Jacobi;

    uint32_t group_count_x_, group_count_y_;
    uint32_t pressure_jacobi_iterations_ = 32;

    // Velocity advection
    lava::texture::s_ptr velocity_field_texture_;
    lava::texture::s_ptr advected_velocity_field_texture_;
    lava::descriptor::s_ptr advect_descriptor_set_layout_;
    VkDescriptorSet advect_descriptor_set_{};
    lava::pipeline_layout::s_ptr advect_pipeline_layout_;
    lava::compute_pipeline::s_ptr advect_pipeline_;

    // Divergence calculation
    lava::texture::s_ptr divergence_field_texture_;
    lava::descriptor::s_ptr divergence_descriptor_set_layout_;
    VkDescriptorSet divergence_descriptor_set_{};
    lava::pipeline_layout::s_ptr divergence_pipeline_layout_;
    lava::compute_pipeline::s_ptr divergence_pipeline_;

    // Pressure calculation using Jacobi iteration
    lava::texture::s_ptr pressure_field_jacobi_texture_A_;
    lava::texture::s_ptr pressure_field_jacobi_texture_B_;
    lava::descriptor::s_ptr pressure_jacobi_descriptor_set_layout_;
    VkDescriptorSet pressure_jacobi_descriptor_set_A_{};
    VkDescriptorSet pressure_jacobi_descriptor_set_B_{};
    lava::pipeline_layout::s_ptr pressure_jacobi_pipeline_layout_;
    lava::compute_pipeline::s_ptr pressure_jacobi_pipeline_;

    // Pressure calculation using unified kernel
    lava::descriptor::s_ptr pressure_kernel_descriptor_set_layout_;
    VkDescriptorSet pressure_kernel_descriptor_set_;
    lava::pipeline_layout::s_ptr pressure_kernel_pipeline_layout_;
    lava::compute_pipeline::s_ptr pressure_kernel_pipeline_;

    // Velocity Update
    lava::texture::s_ptr color_field_texture_A_;
    lava::texture::s_ptr color_field_texture_B_;
    lava::descriptor::s_ptr velocity_update_set_layout_;
    VkDescriptorSet velocity_update_descriptor_set_{};
    lava::pipeline_layout::s_ptr velocity_update_pipeline_layout_;
    lava::compute_pipeline::s_ptr velocity_update_pipeline_;

    // Advect color dye
    lava::descriptor::s_ptr advect_color_set_layout_;
    VkDescriptorSet advect_color_descriptor_set_{};
    lava::pipeline_layout::s_ptr advect_color_pipeline_layout_;
    lava::compute_pipeline::s_ptr advect_color_pipeline_;

    // Update color field
    lava::descriptor::s_ptr color_update_set_layout_;
    VkDescriptorSet color_update_descriptor_set_{};
    lava::pipeline_layout::s_ptr color_update_pipeline_layout_;
    lava::compute_pipeline::s_ptr color_update_pipeline_;

};
}

#endif // SIMULATION_HPP