#pragma once
#ifndef DIVERGENCE_CALCULATION_PASS_HPP
#define DIVERGENCE_CALCULATION_PASS_HPP

#include "ComputePass.hpp"
#include "ResourceManager.hpp"
#include <liblava/lava.hpp>

namespace FluidSimulation
{
class DivergenceCalculationPass : public ComputePass
{
  public:
    using s_ptr = std::shared_ptr<DivergenceCalculationPass>;

    DivergenceCalculationPass(lava::engine &app, lava::descriptor::pool::s_ptr pool);
    ~DivergenceCalculationPass() override;

    void CreateDescriptorSets() override;
    void UpdateDescriptorSets() override;
    void CreatePipeline() override;
    void Execute(VkCommandBuffer cmd_buffer, const SimulationConstants &constants) override;

    static s_ptr Make(lava::engine &app, lava::descriptor::pool::s_ptr pool)
    {
        return std::make_shared<DivergenceCalculationPass>(app, pool);
    }

  private:
    lava::descriptor::s_ptr descriptor_set_layout_;
    VkDescriptorSet descriptor_set_{};

    lava::texture::s_ptr advected_velocity_field_;
    lava::texture::s_ptr divergence_field_;
    lava::texture::s_ptr obstacle_mask_;
};

} // namespace FluidSimulation

#endif // DIVERGENCE_CALCULATION_PASS_HPP