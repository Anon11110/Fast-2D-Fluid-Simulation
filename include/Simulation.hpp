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

struct MultigridConstants
{
    int fine_width;
    int fine_height;
    int coarse_width;
    int coarse_height;
};

enum class PressureProjectionMethod
{
    Jacobi,
    Kernel,
    Multigrid
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
    void CreateMultigridTextures(uint32_t max_levels);
    void CreateDescriptorPool();
    void CreateDescriptorSets();
    void SetupPipelines();
    void UpdateDescriptorSets();
    void JacobiPressureProjection(VkCommandBuffer cmd_buffer, const SimulationConstants &constants);
    void KernelPressureProjection(VkCommandBuffer cmd_buffer, const SimulationConstants &constants);

    void PerformRelaxation(VkCommandBuffer cmd_buffer, const SimulationConstants &constants, uint32_t level);
    void PerformRestriction(VkCommandBuffer cmd_buffer, const MultigridConstants &constants, uint32_t level);
    void PerformProlongation(VkCommandBuffer cmd_buffer, const MultigridConstants &constants, uint32_t level);
    MultigridConstants CalculateMultigridConstants(uint32_t level, uint32_t max_levels);
    void VCyclePressureProjection(VkCommandBuffer cmd_buffer, const SimulationConstants &simulation_constants,
                                  uint32_t max_levels);

    lava::engine &app_;

    lava::descriptor::pool::s_ptr descriptor_pool_;

    bool reset_flag_ = true;

    PressureProjectionMethod pressure_projection_method_ = PressureProjectionMethod::Jacobi;

    uint32_t group_count_x_, group_count_y_;
    uint32_t pressure_jacobi_iterations_ = 32;
    uint32_t multigrid_levels_ = 8;
    uint32_t relaxation_iterations_ = 2;

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

    //// V-Cycle Multigrid
    // Pressure relaxation
    std::vector<lava::texture::s_ptr> pressure_multigrid_texture_A_;
    std::vector<lava::texture::s_ptr> pressure_multigrid_texture_B_;
    lava::descriptor::s_ptr relaxation_descriptor_set_layout_;
    std::vector<VkDescriptorSet> relaxation_descriptor_sets_A_;
    std::vector<VkDescriptorSet> relaxation_descriptor_sets_B_;
    lava::pipeline_layout::s_ptr relaxation_pipeline_layout_;
    lava::compute_pipeline::s_ptr relaxation_pipeline_;

    // Pressure restriction
    std::vector<VkDescriptorSet> restriction_descriptor_sets_;
    lava::descriptor::s_ptr restriction_descriptor_set_layout_;
    lava::pipeline_layout::s_ptr restriction_pipeline_layout_;
    lava::compute_pipeline::s_ptr restriction_pipeline_;

    // Pressure prolongation
    std::vector<VkDescriptorSet> prolongation_descriptor_sets_;
    lava::descriptor::s_ptr prolongation_descriptor_set_layout_;
    lava::pipeline_layout::s_ptr prolongation_pipeline_layout_;
    lava::compute_pipeline::s_ptr prolongation_pipeline_;
    //// V-Cycle Multigrid

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
} // namespace FluidSimulation

#endif // SIMULATION_HPP