#pragma once
#ifndef JACOBI_PRESSURE_PASS_HPP
#define JACOBI_PRESSURE_PASS_HPP

#include "ComputePass.hpp"
#include "ResourceManager.hpp"
#include <liblava/lava.hpp>

namespace FluidSimulation
{
class JacobiPressurePass : public ComputePass
{
  public:
    using s_ptr = std::shared_ptr<JacobiPressurePass>;

    JacobiPressurePass(lava::engine &app, lava::descriptor::pool::s_ptr pool);
    ~JacobiPressurePass() override;

    void CreateDescriptorSets() override;
    void UpdateDescriptorSets() override;
    void CreatePipeline() override;
    void Execute(VkCommandBuffer cmd_buffer, const SimulationConstants &constants) override;

    void SetIterations(uint32_t iterations)
    {
        pressure_jacobi_iterations_ = iterations;
    }
    uint32_t GetIterations() const
    {
        return pressure_jacobi_iterations_;
    }

    static s_ptr Make(lava::engine &app, lava::descriptor::pool::s_ptr pool)
    {
        return std::make_shared<JacobiPressurePass>(app, pool);
    }

  private:
    lava::descriptor::s_ptr descriptor_set_layout_;
    VkDescriptorSet descriptor_set_A_{};
    VkDescriptorSet descriptor_set_B_{};

    lava::texture::s_ptr divergence_field_;
    lava::texture::s_ptr pressure_field_A_;
    lava::texture::s_ptr pressure_field_B_;
    lava::texture::s_ptr obstacle_mask_;

    uint32_t pressure_jacobi_iterations_ = 32;
};

} // namespace FluidSimulation

#endif // JACOBI_PRESSURE_PASS_HPP