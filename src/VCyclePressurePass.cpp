#include "VCyclePressurePass.hpp"

namespace FluidSimulation
{

VCyclePressurePass::VCyclePressurePass(lava::engine &app, lava::descriptor::pool::s_ptr pool, uint32_t max_levels)
    : ComputePass(app, pool), max_levels_(max_levels)
{
    auto &resource_manager = ResourceManager::GetInstance();

    divergence_fields_.resize(max_levels_);
    divergence_fields_[0] = resource_manager.GetTexture("divergence_field");
    for (uint32_t level = 1; level < max_levels_; level++)
    {
        divergence_fields_[level] = resource_manager.GetTexture("residual_L" + std::to_string(level));
    }

    obstacle_mask_ = resource_manager.GetTexture("obstacle_mask");

    pressure_multigrid_texture_A_.resize(max_levels_);
    pressure_multigrid_texture_B_.resize(max_levels_);
    multigrid_temp_textures_.resize(max_levels_);
    multigrid_temp_textures1_.resize(max_levels_);

    // Use existing pressure field for level 0
    pressure_multigrid_texture_A_[0] = resource_manager.GetTexture("pressure_field_A");
    pressure_multigrid_texture_B_[0] = resource_manager.GetTexture("pressure_field_B");

    for (uint32_t level = 0; level < max_levels_; level++)
    {
        std::string suffix = "_L" + std::to_string(level);
        multigrid_temp_textures_[level] = resource_manager.GetTexture("multigrid_temp" + suffix);
        multigrid_temp_textures1_[level] = resource_manager.GetTexture("multigrid_temp1" + suffix);
        if (level > 0)
        {
            pressure_multigrid_texture_A_[level] = resource_manager.GetTexture("pressure_multigrid_A" + suffix);
            pressure_multigrid_texture_B_[level] = resource_manager.GetTexture("pressure_multigrid_B" + suffix);
        }
    }

    CreateDescriptorSets();
    CreatePipeline();
    UpdateDescriptorSets();
}

VCyclePressurePass::~VCyclePressurePass()
{
    if (relaxation_descriptor_set_layout_)
    {
        relaxation_descriptor_set_layout_->destroy();
    }
    if (relaxation_pipeline_layout_)
    {
        relaxation_pipeline_layout_->destroy();
    }
    if (relaxation_pipeline_)
    {
        relaxation_pipeline_->destroy();
    }

    if (poisson_relaxation_descriptor_set_layout_)
    {
        poisson_relaxation_descriptor_set_layout_->destroy();
    }
    if (poisson_relaxation_pipeline_layout_)
    {
        poisson_relaxation_pipeline_layout_->destroy();
    }
    if (poisson_relaxation_pipeline_)
    {
        poisson_relaxation_pipeline_->destroy();
    }

    if (residual_descriptor_set_layout_)
    {
        residual_descriptor_set_layout_->destroy();
    }
    if (residual_pipeline_layout_)
    {
        residual_pipeline_layout_->destroy();
    }
    if (residual_pipeline_)
    {
        residual_pipeline_->destroy();
    }

    if (restriction_descriptor_set_layout_)
    {
        restriction_descriptor_set_layout_->destroy();
    }
    if (restriction_pipeline_layout_)
    {
        restriction_pipeline_layout_->destroy();
    }
    if (restriction_pipeline_)
    {
        restriction_pipeline_->destroy();
    }

    if (prolongation_descriptor_set_layout_)
    {
        prolongation_descriptor_set_layout_->destroy();
    }
    if (prolongation_pipeline_layout_)
    {
        prolongation_pipeline_layout_->destroy();
    }
    if (prolongation_pipeline_)
    {
        prolongation_pipeline_->destroy();
    }
}

void VCyclePressurePass::CreateDescriptorSets()
{
    // Relaxation descriptor sets
    relaxation_descriptor_set_layout_ = lava::descriptor::make();
    relaxation_descriptor_set_layout_->add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                   VK_SHADER_STAGE_COMPUTE_BIT); // Divergence field
    relaxation_descriptor_set_layout_->add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                   VK_SHADER_STAGE_COMPUTE_BIT); // Previous pressure field
    relaxation_descriptor_set_layout_->add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                   VK_SHADER_STAGE_COMPUTE_BIT); // Pressure field
    relaxation_descriptor_set_layout_->add_binding(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                   VK_SHADER_STAGE_COMPUTE_BIT); // Obstacle mask

    if (!relaxation_descriptor_set_layout_->create(app_.device))
    {
        lava::logger()->error("Failed to create relaxation descriptor set layout");
        throw std::runtime_error("Failed to create relaxation descriptor set layout");
    }

    // Poisson relaxation descriptor sets
    poisson_relaxation_descriptor_set_layout_ = lava::descriptor::make();
    poisson_relaxation_descriptor_set_layout_->add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                           VK_SHADER_STAGE_COMPUTE_BIT); // Divergence field
    poisson_relaxation_descriptor_set_layout_->add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                           VK_SHADER_STAGE_COMPUTE_BIT); // Pressure field
    poisson_relaxation_descriptor_set_layout_->add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                           VK_SHADER_STAGE_COMPUTE_BIT); // Temp texture
    poisson_relaxation_descriptor_set_layout_->add_binding(3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                           VK_SHADER_STAGE_COMPUTE_BIT); // Temp texture 1
    poisson_relaxation_descriptor_set_layout_->add_binding(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                           VK_SHADER_STAGE_COMPUTE_BIT); // Obstacle mask

    if (!poisson_relaxation_descriptor_set_layout_->create(app_.device))
    {
        lava::logger()->error("Failed to create poisson relaxation descriptor set layout");
        throw std::runtime_error("Failed to create poisson relaxation descriptor set layout");
    }

    // Residual calculation descriptor sets
    residual_descriptor_set_layout_ = lava::descriptor::make();
    residual_descriptor_set_layout_->add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                 VK_SHADER_STAGE_COMPUTE_BIT); // Divergence texture
    residual_descriptor_set_layout_->add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                 VK_SHADER_STAGE_COMPUTE_BIT); // Pressure texture
    residual_descriptor_set_layout_->add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                 VK_SHADER_STAGE_COMPUTE_BIT); // Residual texture

    if (!residual_descriptor_set_layout_->create(app_.device))
    {
        lava::logger()->error("Failed to create residual descriptor set layout");
        throw std::runtime_error("Failed to create residual descriptor set layout");
    }

    // Restriction descriptor sets
    restriction_descriptor_set_layout_ = lava::descriptor::make();
    restriction_descriptor_set_layout_->add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                    VK_SHADER_STAGE_COMPUTE_BIT); // Fine grid texture
    restriction_descriptor_set_layout_->add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                    VK_SHADER_STAGE_COMPUTE_BIT); // Coarse grid texture

    if (!restriction_descriptor_set_layout_->create(app_.device))
    {
        lava::logger()->error("Failed to create restriction descriptor set layout");
        throw std::runtime_error("Failed to create restriction descriptor set layout");
    }

    // Prolongation descriptor sets
    prolongation_descriptor_set_layout_ = lava::descriptor::make();
    prolongation_descriptor_set_layout_->add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                     VK_SHADER_STAGE_COMPUTE_BIT); // Coarse grid texture
    prolongation_descriptor_set_layout_->add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                     VK_SHADER_STAGE_COMPUTE_BIT); // Fine grid texture

    if (!prolongation_descriptor_set_layout_->create(app_.device))
    {
        lava::logger()->error("Failed to create prolongation descriptor set layout");
        throw std::runtime_error("Failed to create prolongation descriptor set layout");
    }

    // Allocate descriptor sets for each level
    relaxation_descriptor_sets_A_.resize(max_levels_);
    relaxation_descriptor_sets_B_.resize(max_levels_);
    poisson_relaxation_descriptor_sets_.resize(max_levels_);
    residual_descriptor_sets_.resize(max_levels_);
    restriction_descriptor_sets_.resize(max_levels_ - 1);
    prolongation_descriptor_sets_.resize(max_levels_ - 1);

    for (uint32_t level = 0; level < max_levels_; level++)
    {
        relaxation_descriptor_sets_A_[level] = relaxation_descriptor_set_layout_->allocate(descriptor_pool_->get());
        relaxation_descriptor_sets_B_[level] = relaxation_descriptor_set_layout_->allocate(descriptor_pool_->get());
        poisson_relaxation_descriptor_sets_[level] =
            poisson_relaxation_descriptor_set_layout_->allocate(descriptor_pool_->get());
        residual_descriptor_sets_[level] = residual_descriptor_set_layout_->allocate(descriptor_pool_->get());

        if (level < max_levels_ - 1)
        {
            restriction_descriptor_sets_[level] = restriction_descriptor_set_layout_->allocate(descriptor_pool_->get());
            prolongation_descriptor_sets_[level] =
                prolongation_descriptor_set_layout_->allocate(descriptor_pool_->get());
        }
    }
}

void VCyclePressurePass::UpdateDescriptorSets()
{
    for (uint32_t level = 0; level < max_levels_; level++)
    {
        // Update relaxation descriptor sets
        {
            VkDescriptorImageInfo divergence_info{.sampler = VK_NULL_HANDLE,
                                                  .imageView = divergence_fields_[level]->get_image()->get_view(),
                                                  .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

            VkDescriptorImageInfo pressure_A_info{.sampler = VK_NULL_HANDLE,
                                                  .imageView =
                                                      pressure_multigrid_texture_A_[level]->get_image()->get_view(),
                                                  .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

            VkDescriptorImageInfo pressure_B_info{.sampler = VK_NULL_HANDLE,
                                                  .imageView =
                                                      pressure_multigrid_texture_B_[level]->get_image()->get_view(),
                                                  .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

            VkDescriptorImageInfo obstacle_info{.sampler = obstacle_mask_->get_sampler(),
                                                .imageView = obstacle_mask_->get_image()->get_view(),
                                                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

            std::vector<VkDescriptorImageInfo> image_infos_A = {divergence_info, pressure_A_info, pressure_B_info,
                                                                obstacle_info};
            std::vector<VkDescriptorImageInfo> image_infos_B = {divergence_info, pressure_B_info, pressure_A_info,
                                                                obstacle_info};

            std::vector<VkDescriptorType> descriptor_types = {
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER};

            ComputePass::UpdateDescriptorSets(relaxation_descriptor_sets_A_[level], image_infos_A, descriptor_types);
            ComputePass::UpdateDescriptorSets(relaxation_descriptor_sets_B_[level], image_infos_B, descriptor_types);
        }

        // Update Poisson relaxation descriptor sets
        {
            VkDescriptorImageInfo divergence_info{.sampler = VK_NULL_HANDLE,
                                                  .imageView = divergence_fields_[level]->get_image()->get_view(),
                                                  .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

            VkDescriptorImageInfo pressure_info{.sampler = VK_NULL_HANDLE,
                                                .imageView =
                                                    pressure_multigrid_texture_A_[level]->get_image()->get_view(),
                                                .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

            VkDescriptorImageInfo temp_info{.sampler = VK_NULL_HANDLE,
                                            .imageView = multigrid_temp_textures_[level]->get_image()->get_view(),
                                            .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

            VkDescriptorImageInfo temp1_info{.sampler = VK_NULL_HANDLE,
                                             .imageView = multigrid_temp_textures1_[level]->get_image()->get_view(),
                                             .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

            VkDescriptorImageInfo obstacle_info{.sampler = obstacle_mask_->get_sampler(),
                                                .imageView = obstacle_mask_->get_image()->get_view(),
                                                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

            std::vector<VkDescriptorImageInfo> image_infos = {divergence_info, pressure_info, temp_info, temp1_info,
                                                              obstacle_info};

            std::vector<VkDescriptorType> descriptor_types = {
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER};

            ComputePass::UpdateDescriptorSets(poisson_relaxation_descriptor_sets_[level], image_infos,
                                              descriptor_types);
        }

        // Update restriction and prolongation descriptor sets for non-final levels
        if (level < max_levels_ - 1)
        {
            // Residual calculation
            {
                VkDescriptorImageInfo divergence_info{.sampler = VK_NULL_HANDLE,
                                                      .imageView = divergence_fields_[level]->get_image()->get_view(),
                                                      .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

                VkDescriptorImageInfo pressure_info{.sampler = VK_NULL_HANDLE,
                                                    .imageView =
                                                        pressure_multigrid_texture_A_[level]->get_image()->get_view(),
                                                    .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
                VkDescriptorImageInfo residual_info{
                    .sampler = VK_NULL_HANDLE,
                    .imageView = divergence_fields_[level + 1]->get_image()->get_view(), // Residual at next level
                    .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

                std::vector<VkDescriptorImageInfo> image_infos = {divergence_info, pressure_info, residual_info};
                std::vector<VkDescriptorType> descriptor_types = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                                  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                                  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};

                ComputePass::UpdateDescriptorSets(residual_descriptor_sets_[level], image_infos, descriptor_types);
            }

            // Restriction
            VkDescriptorImageInfo fine_grid_divergence_info = {.sampler = VK_NULL_HANDLE,
                                                    .imageView =
                                                        divergence_fields_[level]->get_image()->get_view(),
                                                    .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
            VkDescriptorImageInfo coarse_grid_divergence_info = {
                .sampler = VK_NULL_HANDLE,
                .imageView = divergence_fields_[level + 1]->get_image()->get_view(),
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

            std::vector<VkDescriptorImageInfo> restriction_infos = {fine_grid_divergence_info,
                                                                    coarse_grid_divergence_info};
            std::vector<VkDescriptorType> restriction_types = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                               VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};

            ComputePass::UpdateDescriptorSets(restriction_descriptor_sets_[level], restriction_infos,
                                              restriction_types);

            // Prolongation
            VkDescriptorImageInfo coarse_grid_pressure_info = {
                .sampler = VK_NULL_HANDLE,
                .imageView = pressure_multigrid_texture_A_[level + 1]->get_image()->get_view(),
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
            VkDescriptorImageInfo fine_grid_pressure_info = {.sampler = VK_NULL_HANDLE,
                                                               .imageView = pressure_multigrid_texture_A_[level]->get_image()->get_view(),
                                                               .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
            
            std::vector<VkDescriptorImageInfo> prolongation_infos = {coarse_grid_pressure_info,
                                                                     fine_grid_pressure_info};

            std::vector<VkDescriptorType> prolongation_types = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
            ComputePass::UpdateDescriptorSets(prolongation_descriptor_sets_[level], prolongation_infos,
                                              prolongation_types);
        }
    }
}

void VCyclePressurePass::CreatePipeline()
{
    CreateRelaxationPipeline();
    CreateResidualPipeline();
    CreateRestrictionPipeline();
    CreateProlongationPipeline();
    CreatePoissonRelaxationPipeline();
}

void VCyclePressurePass::CreateBasePipeline(lava::compute_pipeline::s_ptr &pipeline, const char *shader_name,
                                            lava::descriptor::s_ptr descriptor_set_layout,
                                            lava::pipeline_layout::s_ptr &existing_pipeline_layout,
                                            size_t push_constant_size)
{
    pipeline = lava::compute_pipeline::make(app_.device);
    existing_pipeline_layout = lava::pipeline_layout::make();
    existing_pipeline_layout->add(descriptor_set_layout);

    if (push_constant_size > 0)
    {
        existing_pipeline_layout->add_push_constant_range(
            {VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(push_constant_size)});
    }

    if (!existing_pipeline_layout->create(app_.device))
    {
        throw std::runtime_error("Failed to create pipeline layout");
    }

    lava::c_data shader_data = app_.producer.get_shader(shader_name);
    if (!shader_data.addr)
    {
        throw std::runtime_error("Failed to load shader");
    }

    pipeline->set_shader_stage(shader_data, VK_SHADER_STAGE_COMPUTE_BIT);
    pipeline->set_layout(existing_pipeline_layout);

    if (!pipeline->create())
    {
        throw std::runtime_error("Failed to create pipeline");
    }
}

void VCyclePressurePass::CreateRelaxationPipeline()
{
    relaxation_pipeline_ = lava::compute_pipeline::make(app_.device);
    CreateBasePipeline(relaxation_pipeline_, "PressureRelaxation.comp", relaxation_descriptor_set_layout_,
                       relaxation_pipeline_layout_, sizeof(SimulationConstants));
}

void VCyclePressurePass::CreateResidualPipeline()
{
    residual_pipeline_ = lava::compute_pipeline::make(app_.device);
    CreateBasePipeline(residual_pipeline_, "PressureResidualCalculation.comp", residual_descriptor_set_layout_,
                       residual_pipeline_layout_, sizeof(MultigridConstants));
}

void VCyclePressurePass::CreateRestrictionPipeline()
{
    restriction_pipeline_ = lava::compute_pipeline::make(app_.device);
    CreateBasePipeline(restriction_pipeline_, "PressureRestriction.comp", restriction_descriptor_set_layout_,
                       restriction_pipeline_layout_, sizeof(MultigridConstants));
}

void VCyclePressurePass::CreateProlongationPipeline()
{
    prolongation_pipeline_ = lava::compute_pipeline::make(app_.device);
    CreateBasePipeline(prolongation_pipeline_, "PressureProlongation.comp", prolongation_descriptor_set_layout_,
                       prolongation_pipeline_layout_, sizeof(MultigridConstants));
}

void VCyclePressurePass::CreatePoissonRelaxationPipeline()
{
    poisson_relaxation_pipeline_ = lava::compute_pipeline::make(app_.device);
    CreateBasePipeline(poisson_relaxation_pipeline_, "PressureRelaxationPoisson.comp",
                       poisson_relaxation_descriptor_set_layout_, poisson_relaxation_pipeline_layout_,
                       sizeof(SimulationConstants));
}

void VCyclePressurePass::PerformRelaxation(VkCommandBuffer cmd_buffer, const SimulationConstants &constants,
                                           uint32_t level)
{
    // Always track active read/write textures
    lava::texture::s_ptr active_read_texture = pressure_multigrid_texture_A_[level];
    lava::texture::s_ptr active_write_texture = pressure_multigrid_texture_B_[level];

    divergence_fields_[level]->get_image()->transition_layout(
        cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    for (uint32_t i = 0; i < relaxation_iterations_; i++)
    {
        active_read_texture->get_image()->transition_layout(
            cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        active_write_texture->get_image()->transition_layout(
            cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        VkDescriptorSet pressure_descriptor_set =
            (i % 2 == 0) ? relaxation_descriptor_sets_A_[level] : relaxation_descriptor_sets_B_[level];

        relaxation_pipeline_->bind(cmd_buffer);
        vkCmdPushConstants(cmd_buffer, relaxation_pipeline_->get_layout()->get(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(SimulationConstants), &constants);
        vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, relaxation_pipeline_->get_layout()->get(),
                                0, 1, &pressure_descriptor_set, 0, nullptr);

        vkCmdDispatch(cmd_buffer, (constants.texture_width + 15) / 16, (constants.texture_height + 15) / 16, 1);

        // Swap read/write textures for next iteration
        std::swap(active_read_texture, active_write_texture);
    }

    // Ensure final result is in texture A
    std::swap(pressure_multigrid_texture_A_[level], pressure_multigrid_texture_B_[level]);
    std::swap(relaxation_descriptor_sets_A_[level], relaxation_descriptor_sets_B_[level]);
}

void VCyclePressurePass::PerformPoissonFilterRelaxation(VkCommandBuffer cmd_buffer,
                                                        const SimulationConstants &constants, uint32_t level)
{
    divergence_fields_[level]->get_image()->transition_layout(cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT,
                                                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    pressure_multigrid_texture_A_[level]->get_image()->transition_layout(
        cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    multigrid_temp_textures_[level]->get_image()->transition_layout(
        cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    multigrid_temp_textures1_[level]->get_image()->transition_layout(
        cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    poisson_relaxation_pipeline_->bind(cmd_buffer);

    vkCmdPushConstants(cmd_buffer, poisson_relaxation_pipeline_->get_layout()->get(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(SimulationConstants), &constants);

    vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            poisson_relaxation_pipeline_->get_layout()->get(), 0, 1,
                            &poisson_relaxation_descriptor_sets_[level], 0, nullptr);

    vkCmdDispatch(cmd_buffer, (constants.texture_width + 15) / 16, (constants.texture_height + 15) / 16, 1);
}

void VCyclePressurePass::CalculateResidual(VkCommandBuffer cmd_buffer, const MultigridConstants &constants,
                                           uint32_t level)
{
    divergence_fields_[level]->get_image()->transition_layout(
        cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    pressure_multigrid_texture_A_[level]->get_image()->transition_layout(
        cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    divergence_fields_[level + 1]->get_image()->transition_layout(
        cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    residual_pipeline_->bind(cmd_buffer);

    vkCmdPushConstants(cmd_buffer, residual_pipeline_->get_layout()->get(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(MultigridConstants), &constants);

    vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, residual_pipeline_->get_layout()->get(), 0, 1,
                            &residual_descriptor_sets_[level], 0, nullptr);

    vkCmdDispatch(cmd_buffer, (constants.coarse_width + 15) / 16, (constants.coarse_height + 15) / 16, 1);
}


void VCyclePressurePass::PerformRestriction(VkCommandBuffer cmd_buffer, const MultigridConstants &constants,
                                            uint32_t level)
{
    divergence_fields_[level]->get_image()->transition_layout(
        cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    divergence_fields_[level + 1]->get_image()->transition_layout(
        cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    restriction_pipeline_->bind(cmd_buffer);
    vkCmdPushConstants(cmd_buffer, restriction_pipeline_->get_layout()->get(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(MultigridConstants), &constants);
    vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, restriction_pipeline_->get_layout()->get(), 0,
                            1, &restriction_descriptor_sets_[level], 0, nullptr);

    vkCmdDispatch(cmd_buffer, (constants.coarse_width + 15) / 16, (constants.coarse_height + 15) / 16, 1);
}

void VCyclePressurePass::PerformProlongation(VkCommandBuffer cmd_buffer, const MultigridConstants &constants,
                                             uint32_t level)
{
    pressure_multigrid_texture_A_[level + 1]->get_image()->transition_layout(
        cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    pressure_multigrid_texture_A_[level]->get_image()->transition_layout(
        cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    prolongation_pipeline_->bind(cmd_buffer);
    vkCmdPushConstants(cmd_buffer, prolongation_pipeline_->get_layout()->get(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(MultigridConstants), &constants);
    vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, prolongation_pipeline_->get_layout()->get(), 0,
                            1, &prolongation_descriptor_sets_[level], 0, nullptr);
    vkCmdDispatch(cmd_buffer, (constants.fine_width + 15) / 16, (constants.fine_height + 15) / 16, 1);
}

void VCyclePressurePass::Execute(VkCommandBuffer cmd_buffer, const SimulationConstants &constants)
{
    for (int i = 0; i < vcycle_iterations_; i++)
    {
        // Initial relaxation for the finest grid
        if (relaxation_type_ == VCycleRelaxationType::Poisson_Filter)
        {
            PerformPoissonFilterRelaxation(cmd_buffer, constants, 0);
        }
        else
        {
            PerformRelaxation(cmd_buffer, constants, 0);
        }

        SimulationConstants level_constants = constants;

        // Restrict residual to coarser grids
        for (uint32_t level = 0; level < max_levels_ - 1; level++)
        {
            MultigridConstants multigrid_constants = CalculateMultigridConstants(level);
            CalculateResidual(cmd_buffer, multigrid_constants, level);
            PerformRestriction(cmd_buffer, multigrid_constants, level);

            level_constants.texture_width = pressure_multigrid_texture_A_[level + 1]->get_image()->get_size().x;
            level_constants.texture_height = pressure_multigrid_texture_A_[level + 1]->get_image()->get_size().y;

            if (relaxation_type_ == VCycleRelaxationType::Poisson_Filter)
            {
                PerformPoissonFilterRelaxation(cmd_buffer, level_constants, level + 1);
            }
            else
            {
                PerformRelaxation(cmd_buffer, level_constants, level + 1);
            }
        }

        // Prolong correction back to finer grids
        for (int32_t level = max_levels_ - 2; level >= 0; level--)
        {
            MultigridConstants multigrid_constants = CalculateMultigridConstants(level);
            PerformProlongation(cmd_buffer, multigrid_constants, level);

            level_constants.texture_width = pressure_multigrid_texture_A_[level]->get_image()->get_size().x;
            level_constants.texture_height = pressure_multigrid_texture_A_[level]->get_image()->get_size().y;

            if (relaxation_type_ == VCycleRelaxationType::Poisson_Filter)
            {
                PerformPoissonFilterRelaxation(cmd_buffer, level_constants, level);
            }
            else
            {
                PerformRelaxation(cmd_buffer, level_constants, level);
            }
        }
    }
}

MultigridConstants VCyclePressurePass::CalculateMultigridConstants(uint32_t level) const
{
    MultigridConstants constants;
    const glm::uvec2 &window_size = app_.target->get_size();

    constants.fine_width = std::max(1u, window_size.x / (1 << level));
    constants.fine_height = std::max(1u, window_size.y / (1 << level));
    constants.coarse_width = std::max(1u, window_size.x / (1 << (level + 1)));
    constants.coarse_height = std::max(1u, window_size.y / (1 << (level + 1)));

    return constants;
}

} // namespace FluidSimulation