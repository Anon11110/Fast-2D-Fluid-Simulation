#pragma once
#ifndef VELOCITY_UPDATE_PASS_HPP
#define VELOCITY_UPDATE_PASS_HPP

#include "ComputePass.hpp"
#include "ResourceManager.hpp"
#include <liblava/lava.hpp>

namespace FluidSimulation
{
class VelocityUpdatePass : public ComputePass
{
  public:
    using s_ptr = std::shared_ptr<VelocityUpdatePass>;

    VelocityUpdatePass(lava::engine &app, lava::descriptor::pool::s_ptr pool);
    ~VelocityUpdatePass() override;

    void CreateDescriptorSets() override;
    void UpdateDescriptorSets() override;
    void CreatePipeline() override;
    void Execute(VkCommandBuffer cmd_buffer, const SimulationConstants &constants) override;

    static s_ptr Make(lava::engine &app, lava::descriptor::pool::s_ptr pool)
    {
        return std::make_shared<VelocityUpdatePass>(app, pool);
    }

  private:
    lava::descriptor::s_ptr descriptor_set_layout_;
    VkDescriptorSet descriptor_set_{};

    lava::texture::s_ptr pressure_field_;
    lava::texture::s_ptr advected_velocity_field_;
    lava::texture::s_ptr velocity_field_;
    lava::texture::s_ptr obstacle_mask_;
};

} // namespace FluidSimulation
#endif // VELOCITY_UPDATE_PASS_HPP