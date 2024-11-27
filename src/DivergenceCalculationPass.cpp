#include "DivergenceCalculationPass.hpp"

namespace FluidSimulation
{

DivergenceCalculationPass::DivergenceCalculationPass(lava::engine &app, lava::descriptor::pool::s_ptr pool)
    : ComputePass(app, pool)
{
    auto &resource_manager = ResourceManager::GetInstance();

    advected_velocity_field_ = resource_manager.GetTexture("advected_velocity_field");
    divergence_field_ = resource_manager.GetTexture("divergence_field");
    obstacle_mask_ = resource_manager.GetTexture("obstacle_mask");

    CreateDescriptorSets();
    CreatePipeline();
    UpdateDescriptorSets();
}

DivergenceCalculationPass::~DivergenceCalculationPass()
{
    if (descriptor_set_layout_)
    {
        descriptor_set_layout_->destroy();
    }
}

void DivergenceCalculationPass::CreateDescriptorSets()
{
    descriptor_set_layout_ = lava::descriptor::make();
    descriptor_set_layout_->add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                        VK_SHADER_STAGE_COMPUTE_BIT); // Advected velocity field
    descriptor_set_layout_->add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                        VK_SHADER_STAGE_COMPUTE_BIT); // Divergence field
    descriptor_set_layout_->add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                        VK_SHADER_STAGE_COMPUTE_BIT); // Obstacle mask

    if (!descriptor_set_layout_->create(app_.device))
    {
        lava::logger()->error("Failed to create divergence descriptor set layout");
        throw std::runtime_error("Failed to create divergence calculation descriptor set layout");
    }

    descriptor_set_ = descriptor_set_layout_->allocate(descriptor_pool_->get());
    if (!descriptor_set_)
    {
        lava::logger()->error("Failed to allocate divergence calculation descriptor set");
        throw std::runtime_error("Failed to allocate divergence calculation descriptor set");
    }
}

void DivergenceCalculationPass::UpdateDescriptorSets()
{
    VkDescriptorImageInfo advected_velocity_info{.sampler = VK_NULL_HANDLE,
                                                 .imageView = advected_velocity_field_->get_image()->get_view(),
                                                 .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

    VkDescriptorImageInfo divergence_field_info{.sampler = VK_NULL_HANDLE,
                                                .imageView = divergence_field_->get_image()->get_view(),
                                                .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

    VkDescriptorImageInfo obstacle_mask_info{.sampler = obstacle_mask_->get_sampler(),
                                             .imageView = obstacle_mask_->get_image()->get_view(),
                                             .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    std::vector<VkDescriptorImageInfo> image_infos = {advected_velocity_info, divergence_field_info,
                                                      obstacle_mask_info};

    std::vector<VkDescriptorType> descriptor_types = {
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER};

    ComputePass::UpdateDescriptorSets(descriptor_set_, image_infos, descriptor_types);
}

void DivergenceCalculationPass::CreatePipeline()
{
    CreateBasePipeline("DivergenceCalculation.comp", descriptor_set_layout_, sizeof(SimulationConstants));
}

void DivergenceCalculationPass::Execute(VkCommandBuffer cmd_buffer, const SimulationConstants &constants)
{
    advected_velocity_field_->get_image()->transition_layout(
        cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    divergence_field_->get_image()->transition_layout(cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT,
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