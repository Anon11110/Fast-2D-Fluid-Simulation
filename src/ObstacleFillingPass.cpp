#include "ObstacleFillingPass.hpp"

namespace FluidSimulation
{

ObstacleFillingPass::ObstacleFillingPass(lava::engine &app, lava::descriptor::pool::s_ptr pool) : ComputePass(app, pool)
{
    auto &resource_manager = ResourceManager::GetInstance();
    obstacle_mask_ = resource_manager.GetTexture("obstacle_mask");

    CreateDescriptorSets();
    CreatePipeline();
    UpdateDescriptorSets();
}

ObstacleFillingPass::~ObstacleFillingPass()
{
    if (descriptor_set_layout_)
    {
        descriptor_set_layout_->destroy();
    }
}

void ObstacleFillingPass::CreateDescriptorSets()
{
    descriptor_set_layout_ = lava::descriptor::make();
    descriptor_set_layout_->add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                        VK_SHADER_STAGE_COMPUTE_BIT); // Obstacle mask

    if (!descriptor_set_layout_->create(app_.device))
    {
        lava::logger()->error("Failed to create obstacle filling descriptor set layout");
        throw std::runtime_error("Failed to create obstacle filling descriptor set layout");
    }

    descriptor_set_ = descriptor_set_layout_->allocate(descriptor_pool_->get());
    if (!descriptor_set_)
    {
        lava::logger()->error("Failed to allocate obstacle filling descriptor set");
        throw std::runtime_error("Failed to allocate obstacle filling descriptor set");
    }
}

void ObstacleFillingPass::UpdateDescriptorSets()
{
    VkDescriptorImageInfo obstacle_mask_info{.sampler = VK_NULL_HANDLE,
                                             .imageView = obstacle_mask_->get_image()->get_view(),
                                             .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

    std::vector<VkDescriptorImageInfo> image_infos = {obstacle_mask_info};

    std::vector<VkDescriptorType> descriptor_types = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};

    ComputePass::UpdateDescriptorSets(descriptor_set_, image_infos, descriptor_types);
}

void ObstacleFillingPass::CreatePipeline()
{
    CreateBasePipeline("ObstacleMaskFilling.comp", descriptor_set_layout_);
}

void ObstacleFillingPass::Execute(VkCommandBuffer cmd_buffer, const SimulationConstants &constants)
{
    if (!needs_update_)
    {
        return;
    }

    obstacle_mask_->get_image()->transition_layout(cmd_buffer, VK_IMAGE_LAYOUT_GENERAL,
                                                   VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    pipeline_->bind(cmd_buffer);

    vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_->get(), 0, 1, &descriptor_set_,
                            0, nullptr);

    uint32_t group_count_x = (constants.texture_width + 15) / 16;
    uint32_t group_count_y = (constants.texture_height + 15) / 16;
    vkCmdDispatch(cmd_buffer, group_count_x, group_count_y, 1);

    needs_update_ = false;
}

} // namespace FluidSimulation