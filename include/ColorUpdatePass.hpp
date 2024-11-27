#pragma once
#ifndef COLOR_UPDATE_PASS_HPP
#define COLOR_UPDATE_PASS_HPP

#include "ComputePass.hpp"
#include "ResourceManager.hpp"
#include <liblava/lava.hpp>

namespace FluidSimulation
{
class ColorUpdatePass : public ComputePass
{
  public:
    using s_ptr = std::shared_ptr<ColorUpdatePass>;

    ColorUpdatePass(lava::engine &app, lava::descriptor::pool::s_ptr pool);
    ~ColorUpdatePass() override;

    void CreateDescriptorSets() override;
    void UpdateDescriptorSets() override;
    void CreatePipeline() override;
    void Execute(VkCommandBuffer cmd_buffer, const SimulationConstants &constants) override;

    static s_ptr Make(lava::engine &app, lava::descriptor::pool::s_ptr pool)
    {
        return std::make_shared<ColorUpdatePass>(app, pool);
    }

  private:
    lava::descriptor::s_ptr descriptor_set_layout_;
    VkDescriptorSet descriptor_set_{};

    lava::texture::s_ptr color_field_B_;
    lava::texture::s_ptr color_field_A_;
    lava::texture::s_ptr obstacle_mask_;
};

} // namespace FluidSimulation
#endif // COLOR_UPDATE_PASS_HPP