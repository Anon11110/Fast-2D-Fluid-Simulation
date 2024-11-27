#include "JacobiPressurePass.hpp"

namespace FluidSimulation
{

JacobiPressurePass::JacobiPressurePass(lava::engine &app, lava::descriptor::pool::s_ptr pool) : ComputePass(app, pool)
{
    auto &resource_manager = ResourceManager::GetInstance();

    divergence_field_ = resource_manager.GetTexture("divergence_field");
    pressure_field_A_ = resource_manager.GetTexture("pressure_field_A");
    pressure_field_B_ = resource_manager.GetTexture("pressure_field_B");
    obstacle_mask_ = resource_manager.GetTexture("obstacle_mask");

    CreateDescriptorSets();
    CreatePipeline();
    UpdateDescriptorSets();
}

JacobiPressurePass::~JacobiPressurePass()
{
    if (descriptor_set_layout_)
    {
        descriptor_set_layout_->destroy();
    }
}

void JacobiPressurePass::CreateDescriptorSets()
{
    descriptor_set_layout_ = lava::descriptor::make();
    descriptor_set_layout_->add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                        VK_SHADER_STAGE_COMPUTE_BIT); // Divergence field
    descriptor_set_layout_->add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                        VK_SHADER_STAGE_COMPUTE_BIT); // Pressure field A
    descriptor_set_layout_->add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                        VK_SHADER_STAGE_COMPUTE_BIT); // Pressure field B
    descriptor_set_layout_->add_binding(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                        VK_SHADER_STAGE_COMPUTE_BIT); // Obstacle mask

    if (!descriptor_set_layout_->create(app_.device))
    {
        throw std::runtime_error("Failed to create Jacobi pressure projection descriptor set layout");
    }

    descriptor_set_A_ = descriptor_set_layout_->allocate(descriptor_pool_->get());
    descriptor_set_B_ = descriptor_set_layout_->allocate(descriptor_pool_->get());

    if (!descriptor_set_A_ || !descriptor_set_B_)
    {
        throw std::runtime_error("Failed to allocate Jacobi pressure projection descriptor sets");
    }
}

void JacobiPressurePass::UpdateDescriptorSets()
{
    VkDescriptorImageInfo divergence_field_info{.sampler = VK_NULL_HANDLE,
                                                .imageView = divergence_field_->get_image()->get_view(),
                                                .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

    VkDescriptorImageInfo pressure_field_A_info{.sampler = VK_NULL_HANDLE,
                                                .imageView = pressure_field_A_->get_image()->get_view(),
                                                .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

    VkDescriptorImageInfo pressure_field_B_info{.sampler = VK_NULL_HANDLE,
                                                .imageView = pressure_field_B_->get_image()->get_view(),
                                                .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

    VkDescriptorImageInfo obstacle_mask_info{.sampler = obstacle_mask_->get_sampler(),
                                             .imageView = obstacle_mask_->get_image()->get_view(),
                                             .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    // Update set A
    std::vector<VkDescriptorImageInfo> image_infos_A = {divergence_field_info, pressure_field_A_info,
                                                        pressure_field_B_info, obstacle_mask_info};

    std::vector<VkDescriptorType> descriptor_types = {
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER};

    ComputePass::UpdateDescriptorSets(descriptor_set_A_, image_infos_A, descriptor_types);

    // Update set B
    std::vector<VkDescriptorImageInfo> image_infos_B = {divergence_field_info, pressure_field_B_info,
                                                        pressure_field_A_info, obstacle_mask_info};

    ComputePass::UpdateDescriptorSets(descriptor_set_B_, image_infos_B, descriptor_types);
}

void JacobiPressurePass::CreatePipeline()
{
    CreateBasePipeline("PressureProjectionJacobi.comp", descriptor_set_layout_, sizeof(SimulationConstants));
}

void JacobiPressurePass::Execute(VkCommandBuffer cmd_buffer, const SimulationConstants &constants)
{
    divergence_field_->get_image()->transition_layout(cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT,
                                                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    pipeline_->bind(cmd_buffer);

    vkCmdPushConstants(cmd_buffer, pipeline_layout_->get(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SimulationConstants),
                       &constants);

    for (uint32_t i = 0; i < pressure_jacobi_iterations_; i++)
    {
        uint32_t phase = i % 2;

        pressure_field_A_->get_image()->transition_layout(
            cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, phase ? VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        pressure_field_B_->get_image()->transition_layout(
            cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, phase ? VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        VkDescriptorSet active_set = phase ? descriptor_set_B_ : descriptor_set_A_;
        vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_->get(), 0, 1, &active_set,
                                0, nullptr);

        uint32_t group_count_x = (constants.texture_width + 15) / 16;
        uint32_t group_count_y = (constants.texture_height + 15) / 16;
        vkCmdDispatch(cmd_buffer, group_count_x, group_count_y, 1);
    }

    // Swap textures by handles since the final result is always stored in texture A
    if (pressure_jacobi_iterations_ % 2 != 0)
    {
        std::swap(pressure_field_A_, pressure_field_B_);
    }
}

} // namespace FluidSimulation