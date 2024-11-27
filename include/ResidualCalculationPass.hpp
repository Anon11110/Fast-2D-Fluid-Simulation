#pragma once
#ifndef RESIDUAL_CALCULATION_PASS_HPP
#define RESIDUAL_CALCULATION_PASS_HPP

#include "ComputePass.hpp"
#include "FluidConstants.hpp"
#include "ResourceManager.hpp"
#include <liblava/lava.hpp>
#include <unordered_map>

namespace FluidSimulation
{
class ResidualCalculationPass : public ComputePass
{
  public:
    using s_ptr = std::shared_ptr<ResidualCalculationPass>;

    ResidualCalculationPass(lava::engine &app, lava::descriptor::pool::s_ptr pool);
    ~ResidualCalculationPass() override;

    void CreateDescriptorSets() override;
    void UpdateDescriptorSets() override;
    void CreatePipeline() override;
    void Execute(VkCommandBuffer cmd_buffer, const SimulationConstants &constants) override;

    void CopyResidualToCPU(VkCommandBuffer cmd_buffer, lava::buffer::s_ptr staging_buffer);

    static s_ptr Make(lava::engine &app, lava::descriptor::pool::s_ptr pool)
    {
        return std::make_shared<ResidualCalculationPass>(app, pool);
    }

  private:
    lava::descriptor::s_ptr descriptor_set_layout_;
    VkDescriptorSet descriptor_set_{};

    lava::texture::s_ptr divergence_field_;
    lava::texture::s_ptr pressure_field_;
    lava::texture::s_ptr residual_texture_;
};

} // namespace FluidSimulation
#endif // RESIDUAL_CALCULATION_PASS_HPP