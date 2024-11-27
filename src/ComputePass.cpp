#include "ComputePass.hpp"

namespace FluidSimulation
{

ComputePass::ComputePass(lava::engine &app, lava::descriptor::pool::s_ptr pool) : app_(app), descriptor_pool_(pool)
{
}

ComputePass::~ComputePass()
{
    if (pipeline_)
        pipeline_->destroy();
    if (pipeline_layout_)
        pipeline_layout_->destroy();
}

void ComputePass::CreateBasePipeline(const char *shader_name, lava::descriptor::s_ptr descriptor_set_layout,
                                     size_t push_constant_size)
{
    pipeline_ = lava::compute_pipeline::make(app_.device);
    pipeline_layout_ = lava::pipeline_layout::make();
    pipeline_layout_->add(descriptor_set_layout);

    if (push_constant_size > 0)
    {
        pipeline_layout_->add_push_constant_range(
            {VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(push_constant_size)});
    }

    if (!pipeline_layout_->create(app_.device))
    {
        throw std::runtime_error("Failed to create pipeline layout");
    }

    lava::c_data shader_data = app_.producer.get_shader(shader_name);
    if (!shader_data.addr)
    {
        throw std::runtime_error("Failed to load shader");
    }

    pipeline_->set_shader_stage(shader_data, VK_SHADER_STAGE_COMPUTE_BIT);
    pipeline_->set_layout(pipeline_layout_);

    if (!pipeline_->create())
    {
        throw std::runtime_error("Failed to create pipeline");
    }
}

void ComputePass::UpdateDescriptorSets(VkDescriptorSet descriptor_set,
                                       const std::vector<VkDescriptorImageInfo> &image_infos,
                                       const std::vector<VkDescriptorType> &descriptor_types)
{
    std::vector<VkWriteDescriptorSet> write_sets;
    write_sets.reserve(image_infos.size());

    for (size_t i = 0; i < image_infos.size(); i++)
    {
        write_sets.push_back({.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                              .dstSet = descriptor_set,
                              .dstBinding = static_cast<uint32_t>(i),
                              .descriptorCount = 1,
                              .descriptorType = descriptor_types[i],
                              .pImageInfo = &image_infos[i]});
    }

    app_.device->vkUpdateDescriptorSets(static_cast<uint32_t>(write_sets.size()), write_sets.data(), 0, nullptr);
}

} // namespace FluidSimulation