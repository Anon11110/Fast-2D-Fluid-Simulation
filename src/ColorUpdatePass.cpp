#include "ColorUpdatePass.hpp"

namespace FluidSimulation
{

ColorUpdatePass::ColorUpdatePass(lava::engine &app, lava::descriptor::pool::s_ptr pool) : ComputePass(app, pool)
{
    auto &resource_manager = ResourceManager::GetInstance();

    color_field_B_ = resource_manager.GetTexture("color_field_B");
    color_field_A_ = resource_manager.GetTexture("color_field_A");
    obstacle_mask_ = resource_manager.GetTexture("obstacle_mask");

    CreateDescriptorSets();
    CreatePipeline();
    UpdateDescriptorSets();
}

ColorUpdatePass::~ColorUpdatePass()
{
    if (descriptor_set_layout_)
    {
        descriptor_set_layout_->destroy();
    }
}

void ColorUpdatePass::CreateDescriptorSets()
{
    descriptor_set_layout_ = lava::descriptor::make();
    descriptor_set_layout_->add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                        VK_SHADER_STAGE_COMPUTE_BIT); // Temporary color field
    descriptor_set_layout_->add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                        VK_SHADER_STAGE_COMPUTE_BIT); // Target color field
    descriptor_set_layout_->add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                        VK_SHADER_STAGE_COMPUTE_BIT); // Obstacle mask

    if (!descriptor_set_layout_->create(app_.device))
    {
        lava::logger()->error("Failed to create color update descriptor set layout");
        throw std::runtime_error("Failed to create color update descriptor set layout");
    }

    descriptor_set_ = descriptor_set_layout_->allocate(descriptor_pool_->get());
    if (!descriptor_set_)
    {
        lava::logger()->error("Failed to allocate color update descriptor set");
        throw std::runtime_error("Failed to allocate color update descriptor set");
    }
}

void ColorUpdatePass::UpdateDescriptorSets()
{
    VkDescriptorImageInfo color_field_B_info{.sampler = VK_NULL_HANDLE,
                                             .imageView = color_field_B_->get_image()->get_view(),
                                             .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

    VkDescriptorImageInfo color_field_A_info{.sampler = VK_NULL_HANDLE,
                                             .imageView = color_field_A_->get_image()->get_view(),
                                             .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

    VkDescriptorImageInfo obstacle_mask_info{.sampler = obstacle_mask_->get_sampler(),
                                             .imageView = obstacle_mask_->get_image()->get_view(),
                                             .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    std::vector<VkDescriptorImageInfo> image_infos = {color_field_B_info, color_field_A_info, obstacle_mask_info};

    std::vector<VkDescriptorType> descriptor_types = {
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER};

    ComputePass::UpdateDescriptorSets(descriptor_set_, image_infos, descriptor_types);
}

void ColorUpdatePass::CreatePipeline()
{
    CreateBasePipeline("ColorUpdate.comp", descriptor_set_layout_, sizeof(SimulationConstants));
}

void ColorUpdatePass::Execute(VkCommandBuffer cmd_buffer, const SimulationConstants &constants)
{
    color_field_A_->get_image()->transition_layout(cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT,
                                                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    color_field_B_->get_image()->transition_layout(cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT,
                                                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    obstacle_mask_->get_image()->transition_layout(cmd_buffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                   VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    pipeline_->bind(cmd_buffer);

    vkCmdPushConstants(cmd_buffer, pipeline_layout_->get(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SimulationConstants),
                       &constants);

    vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_->get(), 0, 1, &descriptor_set_,
                            0, nullptr);

    uint32_t group_count_x = (constants.texture_width + 15) / 16;
    uint32_t group_count_y = (constants.texture_height + 15) / 16;
    vkCmdDispatch(cmd_buffer, group_count_x, group_count_y, 1);
}

} // namespace FluidSimulation