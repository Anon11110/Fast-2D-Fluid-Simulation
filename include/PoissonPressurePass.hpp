#pragma once
#ifndef POISSON_PRESSURE_PASS_HPP
#define POISSON_PRESSURE_PASS_HPP

#include "ComputePass.hpp"
#include "ResourceManager.hpp"
#include <liblava/lava.hpp>

namespace FluidSimulation
{
class PoissonPressurePass : public ComputePass
{
  public:
    using s_ptr = std::shared_ptr<PoissonPressurePass>;

    PoissonPressurePass(lava::engine &app, lava::descriptor::pool::s_ptr pool);
    ~PoissonPressurePass() override;

    void CreateDescriptorSets() override;
    void UpdateDescriptorSets() override;
    void CreatePipeline() override;
    void Execute(VkCommandBuffer cmd_buffer, const SimulationConstants &constants) override;

    static s_ptr Make(lava::engine &app, lava::descriptor::pool::s_ptr pool)
    {
        return std::make_shared<PoissonPressurePass>(app, pool);
    }

  private:
    lava::descriptor::s_ptr descriptor_set_layout_;
    VkDescriptorSet descriptor_set_{};

    lava::texture::s_ptr divergence_field_;
    lava::texture::s_ptr pressure_field_;
    lava::texture::s_ptr temp_texture_;
    lava::texture::s_ptr temp_texture1_;
    lava::texture::s_ptr obstacle_mask_;
};

} // namespace FluidSimulation

#endif // POISSON_PRESSURE_PASS_HPP
