#pragma once
#ifndef VELOCITY_ADVECTION_PASS_HPP
#define VELOCITY_ADVECTION_PASS_HPP

#include "ComputePass.hpp"
#include "ResourceManager.hpp"
#include <liblava/lava.hpp>

namespace FluidSimulation
{
class VelocityAdvectionPass : public ComputePass
{
  public:
    using s_ptr = std::shared_ptr<VelocityAdvectionPass>;

    VelocityAdvectionPass(lava::engine &app, lava::descriptor::pool::s_ptr pool);
    ~VelocityAdvectionPass() override;

    void CreateDescriptorSets() override;
    void UpdateDescriptorSets() override;
    void CreatePipeline() override;
    void Execute(VkCommandBuffer cmd_buffer, const SimulationConstants &constants) override;

    static s_ptr Make(lava::engine &app, lava::descriptor::pool::s_ptr pool)
    {
        return std::make_shared<VelocityAdvectionPass>(app, pool);
    }

  private:
    lava::descriptor::s_ptr descriptor_set_layout_;
    VkDescriptorSet descriptor_set_{};

    lava::texture::s_ptr velocity_field_;
    lava::texture::s_ptr advected_field_;
    lava::texture::s_ptr obstacle_mask_;
};

} // namespace FluidSimulation

#endif // VELOCITY_ADVECTION_PASS_HPP