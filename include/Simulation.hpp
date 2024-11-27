#pragma once

#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include "ColorAdvectPass.hpp"
#include "ColorUpdatePass.hpp"
#include "ComputePass.hpp"
#include "DivergenceCalculationPass.hpp"
#include "JacobiPressurePass.hpp"
#include "ObstacleFillingPass.hpp"
#include "PoissonPressurePass.hpp"
#include "ResidualCalculationPass.hpp"
#include "ResourceManager.hpp"
#include "VCyclePressurePass.hpp"
#include "VelocityAdvectionPass.hpp"
#include "VelocityUpdatePass.hpp"
#include "imgui.h"
#include "liblava/lava.hpp"

namespace FluidSimulation
{

class Simulation
{
  public:
    using s_ptr = std::shared_ptr<Simulation>;

    explicit Simulation(lava::engine &app);
    Simulation(const Simulation &) = delete;
    Simulation &operator=(const Simulation &) = delete;
    Simulation(Simulation &&other) = delete;
    Simulation &operator=(Simulation &&other) = delete;

    ~Simulation();

    void OnUpdate(VkCommandBuffer cmd_buffer, const FrameTimeInfo &frame_context);

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

    void Reset()
    {
        reset_flag_ = true;
    }

    static s_ptr Make(lava::engine &app)
    {
        return std::make_shared<Simulation>(app);
    }

    friend class FluidRenderer;

  private:
    void AddShaderMappings();
    void CreateMultigridTextures(uint32_t max_levels);
    void CreateTextures();
    void CreateBuffers();
    void CreateDescriptorPool();
    void CreateComputePasses();

    lava::engine &app_;

    lava::descriptor::pool::s_ptr descriptor_pool_;

    bool reset_flag_ = true;
    bool calculate_residual_error_ = true;
    bool upload_obstacle_mask_ = true;

    const uint32_t PRESSURE_CONVERGENCE_CHECK_FRAME = 1000;
    uint32_t frame_count_ = 0;
    std::vector<float> residual_host_data_;

    PressureProjectionMethod pressure_projection_method_ = PressureProjectionMethod::Jacobi;

    uint32_t pressure_jacobi_iterations_ = 32;
    uint32_t multigrid_levels_ = 5;
    uint32_t relaxation_iterations_ = 4;

    ObstacleFillingPass::s_ptr obstacle_filling_pass_;
    VelocityAdvectionPass::s_ptr velocity_advect_pass_;
    DivergenceCalculationPass::s_ptr divergence_calculation_pass_;
    JacobiPressurePass::s_ptr jacobi_pressure_projection_pass_;
    PoissonPressurePass::s_ptr poisson_pressure_projection_pass_;
    VCyclePressurePass::s_ptr v_cycle_pressure_projection_pass_;
    VelocityUpdatePass::s_ptr velocity_update_pass_;
    ColorAdvectPass::s_ptr color_advect_pass_;
    ColorUpdatePass::s_ptr color_update_pass_;
    ResidualCalculationPass::s_ptr residual_calculation_pass_;
};
} // namespace FluidSimulation

#endif // SIMULATION_HPP