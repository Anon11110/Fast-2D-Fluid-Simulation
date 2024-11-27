#pragma once
#ifndef COMPUTE_PASS_HPP
#define COMPUTE_PASS_HPP

#include "FluidConstants.hpp"
#include <liblava/lava.hpp>

namespace FluidSimulation
{

class ComputePass
{
  public:
    using s_ptr = std::shared_ptr<ComputePass>;

    explicit ComputePass(lava::engine &app, lava::descriptor::pool::s_ptr pool);
    virtual ~ComputePass();

    ComputePass(const ComputePass &) = delete;
    ComputePass &operator=(const ComputePass &) = delete;
    ComputePass(ComputePass &&) = delete;
    ComputePass &operator=(ComputePass &&) = delete;

    virtual void CreateDescriptorSets() = 0;
    virtual void UpdateDescriptorSets() = 0;
    virtual void CreatePipeline() = 0;
    virtual void Execute(VkCommandBuffer cmd_buffer, const SimulationConstants &constants) = 0;

  protected:
    lava::engine &app_;
    lava::descriptor::pool::s_ptr descriptor_pool_;
    lava::compute_pipeline::s_ptr pipeline_;
    lava::pipeline_layout::s_ptr pipeline_layout_;

    void CreateBasePipeline(const char *shader_name, lava::descriptor::s_ptr descriptor_set_layout,
                            size_t push_constant_size = 0);

    void UpdateDescriptorSets(VkDescriptorSet descriptor_set, const std::vector<VkDescriptorImageInfo> &image_infos,
                              const std::vector<VkDescriptorType> &descriptor_types);
};

} // namespace FluidSimulation
#endif // COMPUTE_PASS_HPP