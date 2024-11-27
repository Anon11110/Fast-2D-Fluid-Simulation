#include "ResidualCalculationPass.hpp"

namespace FluidSimulation
{

ResidualCalculationPass::ResidualCalculationPass(lava::engine &app, lava::descriptor::pool::s_ptr pool)
    : ComputePass(app, pool)
{
    auto &resource_manager = ResourceManager::GetInstance();

    divergence_field_ = resource_manager.GetTexture("divergence_field");
    pressure_field_ = resource_manager.GetTexture("pressure_field_A");
    residual_texture_ = resource_manager.GetTexture("residual");

    CreateDescriptorSets();
    CreatePipeline();
    UpdateDescriptorSets();
}

ResidualCalculationPass::~ResidualCalculationPass()
{
    if (descriptor_set_layout_)
    {
        descriptor_set_layout_->destroy();
        descriptor_set_layout_.reset();
    }
}

void ResidualCalculationPass::CreateDescriptorSets()
{
    descriptor_set_layout_ = lava::descriptor::make();
    descriptor_set_layout_->add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                        VK_SHADER_STAGE_COMPUTE_BIT); // Divergence field
    descriptor_set_layout_->add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                        VK_SHADER_STAGE_COMPUTE_BIT); // Pressure field
    descriptor_set_layout_->add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                        VK_SHADER_STAGE_COMPUTE_BIT); // Residual error

    if (!descriptor_set_layout_->create(app_.device))
    {
        throw std::runtime_error("Failed to create residual calculation descriptor set layout");
    }

    descriptor_set_ = descriptor_set_layout_->allocate(descriptor_pool_->get());
    if (!descriptor_set_)
    {
        lava::logger()->error("Failed to allocate residual calculation descriptor set");
        throw std::runtime_error("Failed to allocate residual calculation descriptor set");
    }
}

void ResidualCalculationPass::UpdateDescriptorSets()
{
    VkDescriptorImageInfo divergence_info{.sampler = VK_NULL_HANDLE,
                                          .imageView = divergence_field_->get_image()->get_view(),
                                          .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

    VkDescriptorImageInfo pressure_info{.sampler = VK_NULL_HANDLE,
                                        .imageView = pressure_field_->get_image()->get_view(),
                                        .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

    VkDescriptorImageInfo residual_info{.sampler = VK_NULL_HANDLE,
                                        .imageView = residual_texture_->get_image()->get_view(),
                                        .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

    std::vector<VkDescriptorType> descriptor_types = {
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};

    std::vector<VkDescriptorImageInfo> image_infos = {divergence_info, pressure_info, residual_info};

    ComputePass::UpdateDescriptorSets(descriptor_set_, image_infos, descriptor_types);
}

void ResidualCalculationPass::CreatePipeline()
{
    CreateBasePipeline("ResidualErrorCalculation.comp", descriptor_set_layout_, sizeof(SimulationConstants));
}

void ResidualCalculationPass::Execute(VkCommandBuffer cmd_buffer, const SimulationConstants &constants)
{
    divergence_field_->get_image()->transition_layout(cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT,
                                                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    pressure_field_->get_image()->transition_layout(cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT,
                                                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    residual_texture_->get_image()->transition_layout(cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT,
                                                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    pipeline_->bind(cmd_buffer);

    vkCmdPushConstants(cmd_buffer, pipeline_layout_->get(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SimulationConstants),
                       &constants);

    vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_->get(), 0, 1, &descriptor_set_,
                            0, nullptr);

    uint32_t group_count_x = (constants.texture_width + 15) / 16;
    uint32_t group_count_y = (constants.texture_height + 15) / 16;
    vkCmdDispatch(cmd_buffer, group_count_x, group_count_y, 1);
}

void ResidualCalculationPass::CopyResidualToCPU(VkCommandBuffer cmd_buffer, lava::buffer::s_ptr staging_buffer)
{
    auto image = residual_texture_->get_image();
    VkDeviceSize image_size = image->get_size().x * image->get_size().y * sizeof(float);

    residual_texture_->get_image()->transition_layout(cmd_buffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                                      VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkBufferImageCopy copy_region = {};
    copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy_region.imageSubresource.mipLevel = 0;
    copy_region.imageSubresource.baseArrayLayer = 0;
    copy_region.imageSubresource.layerCount = 1;
    copy_region.imageExtent = {residual_texture_->get_size().x, residual_texture_->get_size().y, 1};

    vkCmdCopyImageToBuffer(cmd_buffer, image->get(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging_buffer->get(), 1,
                           &copy_region);

    lava::logger()->info("Block copy to CPU executed.");
}

} // namespace FluidSimulation