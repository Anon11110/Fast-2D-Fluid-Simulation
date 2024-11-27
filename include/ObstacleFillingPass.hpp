#pragma once
#ifndef OBSTACLE_FILLING_PASS_HPP
#define OBSTACLE_FILLING_PASS_HPP

#include "ComputePass.hpp"
#include "ResourceManager.hpp"
#include <liblava/lava.hpp>

namespace FluidSimulation
{
class ObstacleFillingPass : public ComputePass
{
  public:
    using s_ptr = std::shared_ptr<ObstacleFillingPass>;

    ObstacleFillingPass(lava::engine &app, lava::descriptor::pool::s_ptr pool);
    ~ObstacleFillingPass() override;

    void CreateDescriptorSets() override;
    void UpdateDescriptorSets() override;
    void CreatePipeline() override;
    void Execute(VkCommandBuffer cmd_buffer, const SimulationConstants &constants) override;

    void SetNeedsUpdate(bool value)
    {
        needs_update_ = value;
    }
    bool GetNeedsUpdate() const
    {
        return needs_update_;
    }

    static s_ptr Make(lava::engine &app, lava::descriptor::pool::s_ptr pool)
    {
        return std::make_shared<ObstacleFillingPass>(app, pool);
    }

  private:
    lava::descriptor::s_ptr descriptor_set_layout_;
    VkDescriptorSet descriptor_set_{};

    lava::texture::s_ptr obstacle_mask_;
    bool needs_update_ = true;
};

} // namespace FluidSimulation
#endif // OBSTACLE_FILLING_PASS_HPP