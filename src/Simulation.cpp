#include "Simulation.hpp"

namespace FluidSimulation
{
Simulation::Simulation(lava::engine &app) : app_(app)
{
    const lava::uv2 window_size = app_.target->get_size();
    group_count_x_ = (window_size.x - 1) / 16 + 1;
    group_count_y_ = (window_size.y - 1) / 16 + 1;
    AddShaderMappings();
    CreateTextures();
    CreateDescriptorPool();
    CreateDescriptorSets();
    SetupPipelines();
    UpdateDescriptorSets();
}

Simulation::~Simulation()
{
    if (descriptor_pool_)
        descriptor_pool_->destroy();

    // Velocity advection
    if (velocity_field_texture_)
        velocity_field_texture_->destroy();
    if (advected_velocity_field_texture_)
        advected_velocity_field_texture_->destroy();
    if (advect_descriptor_set_layout_)
        advect_descriptor_set_layout_->destroy();
    if (advect_pipeline_layout_)
        advect_pipeline_layout_->destroy();
    if (advect_pipeline_)
        advect_pipeline_->destroy();

    // Divergence calculation
    if (divergence_field_texture_)
        divergence_field_texture_->destroy();
    if (divergence_descriptor_set_layout_)
        divergence_descriptor_set_layout_->destroy();
    if (divergence_pipeline_layout_)
        divergence_pipeline_layout_->destroy();
    if (divergence_pipeline_)
        divergence_pipeline_->destroy();

    // Pressure calculation using Jacobi iteration
    if (pressure_field_jacobi_texture_A_)
        pressure_field_jacobi_texture_A_->destroy();
    if (pressure_field_jacobi_texture_B_)
        pressure_field_jacobi_texture_B_->destroy();
    if (pressure_jacobi_descriptor_set_layout_)
        pressure_jacobi_descriptor_set_layout_->destroy();
    if (pressure_jacobi_pipeline_layout_)
        pressure_jacobi_pipeline_layout_->destroy();
    if (pressure_jacobi_pipeline_)
        pressure_jacobi_pipeline_->destroy();

    // Pressure calculation using unified kernel
    if (pressure_kernel_descriptor_set_layout_)
        pressure_kernel_descriptor_set_layout_->destroy();
    if (pressure_kernel_pipeline_layout_)
        pressure_kernel_pipeline_layout_->destroy();
    if (pressure_kernel_pipeline_)
        pressure_kernel_pipeline_->destroy();

    // Velocity update
    if (color_field_texture_A_)
        color_field_texture_A_->destroy();
    if (color_field_texture_B_)
        color_field_texture_B_->destroy();
    if (velocity_update_set_layout_)
        velocity_update_set_layout_->destroy();
    if (velocity_update_pipeline_layout_)
        velocity_update_pipeline_layout_->destroy();
    if (velocity_update_pipeline_)
        velocity_update_pipeline_->destroy();

    // Advect color
    if (advect_color_set_layout_)
        advect_color_set_layout_->destroy();
    if (advect_color_pipeline_layout_)
        advect_color_pipeline_layout_->destroy();
    if (advect_color_pipeline_)
        advect_color_pipeline_->destroy();

    // Update color field
    if (color_update_set_layout_)
        color_update_set_layout_->destroy();
    if (color_update_pipeline_layout_)
        color_update_pipeline_layout_->destroy();
    if (color_update_pipeline_)
        color_update_pipeline_->destroy();
}

void Simulation::AddShaderMappings()
{
    const std::vector<std::pair<std::string, std::string>> file_mappings{
        {"VelocityAdvection.comp", "../shaders/VelocityAdvection.comp"},

        {"DivergenceCalculation.comp", "../shaders/DivergenceCalculation.comp"},

        {"PressureProjectionJacobi.comp", "../shaders/PressureProjectionJacobi.comp"},

        {"PressureProjectionKernel.comp", "../shaders/PressureProjectionKernel.comp"},

        {"VelocityUpdate.comp", "../shaders/VelocityUpdate.comp"},

        {"ColorAdvection.comp", "../shaders/ColorAdvection.comp"},

        {"ColorUpdate.comp", "../shaders/ColorUpdate.comp"},
    };

    for (auto &&[name, file] : file_mappings)
    {
        app_.props.add(name, file);
    }
}

void Simulation::CreateTextures()
{
    const glm::uvec2 window_size = app_.target->get_size();

    auto create_texture =
        [&](lava::texture::s_ptr &texture, VkFormat format, VkImageUsageFlags usage, VkSamplerAddressMode address_mode)
    {
        texture = lava::texture::make();
        if (!texture->create(app_.device, window_size, format, {}, lava::texture_type::tex_2d, address_mode, usage))
        {
            throw std::runtime_error("Failed to create texture.");
        }
    };

    create_texture(velocity_field_texture_, VK_FORMAT_R16G16_SFLOAT,
                   VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT);
    create_texture(advected_velocity_field_texture_, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT, {});
    create_texture(divergence_field_texture_, VK_FORMAT_R16_SFLOAT,
                   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, {});
    create_texture(pressure_field_jacobi_texture_A_, VK_FORMAT_R16_SFLOAT,
                   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                   VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    create_texture(pressure_field_jacobi_texture_B_, VK_FORMAT_R16_SFLOAT,
                   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, {});
    create_texture(color_field_texture_A_, VK_FORMAT_R8G8B8A8_UNORM,
                   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT);
    create_texture(color_field_texture_B_, VK_FORMAT_R8G8B8A8_UNORM,
                   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT);
}

void Simulation::CreateDescriptorPool()
{
    descriptor_pool_ = lava::descriptor::pool::make();
    descriptor_pool_->create(
        app_.device, {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 100}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 100}}, 100);
}

void Simulation::CreateDescriptorSets()
{
    // Velocity advection
    {
        advect_descriptor_set_layout_ = lava::descriptor::make();
        advect_descriptor_set_layout_->add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                   VK_SHADER_STAGE_COMPUTE_BIT); // Velocity field
        advect_descriptor_set_layout_->add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                   VK_SHADER_STAGE_COMPUTE_BIT); // Advected velocity field

        if (!advect_descriptor_set_layout_->create(app_.device))
        {
            lava::logger()->error("Failed to create advect descriptor set layout.");
            throw std::runtime_error("Failed to create advect descriptor set layout.");
        }

        advect_descriptor_set_ = advect_descriptor_set_layout_->allocate(descriptor_pool_->get());
    }

    // Divergence calculation
    {
        divergence_descriptor_set_layout_ = lava::descriptor::make();
        divergence_descriptor_set_layout_->add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                       VK_SHADER_STAGE_COMPUTE_BIT); // Advected velocity field
        divergence_descriptor_set_layout_->add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                       VK_SHADER_STAGE_COMPUTE_BIT); // Divergence field

        if (!divergence_descriptor_set_layout_->create(app_.device))
        {
            lava::logger()->error("Failed to create divergence descriptor set layout.");
            throw std::runtime_error("Failed to create divergence descriptor set layout.");
        }

        divergence_descriptor_set_ = divergence_descriptor_set_layout_->allocate(descriptor_pool_->get());
    }

    // Pressure calculation using Jacobi iteration
    {
        pressure_jacobi_descriptor_set_layout_ = lava::descriptor::make();
        pressure_jacobi_descriptor_set_layout_->add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                     VK_SHADER_STAGE_COMPUTE_BIT); // Divergence field
        pressure_jacobi_descriptor_set_layout_->add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                     VK_SHADER_STAGE_COMPUTE_BIT); // Pressure field A
        pressure_jacobi_descriptor_set_layout_->add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                     VK_SHADER_STAGE_COMPUTE_BIT); // Pressure field B

        if (!pressure_jacobi_descriptor_set_layout_->create(app_.device))
        {
            lava::logger()->error("Failed to create pressure descriptor set layout.");
            throw std::runtime_error("Failed to create pressure descriptor set layout.");
        }

        pressure_jacobi_descriptor_set_A_ = pressure_jacobi_descriptor_set_layout_->allocate(descriptor_pool_->get());
        pressure_jacobi_descriptor_set_B_ = pressure_jacobi_descriptor_set_layout_->allocate(descriptor_pool_->get());
    }

    // Pressure calculation using unified kernel
    {
        pressure_kernel_descriptor_set_layout_ = lava::descriptor::make();
        pressure_kernel_descriptor_set_layout_->add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                            VK_SHADER_STAGE_COMPUTE_BIT); // Divergence field
        pressure_kernel_descriptor_set_layout_->add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                            VK_SHADER_STAGE_COMPUTE_BIT); // Pressure field

        if (!pressure_kernel_descriptor_set_layout_->create(app_.device))
        {
            lava::logger()->error("Failed to create pressure descriptor set layout.");
            throw std::runtime_error("Failed to create pressure descriptor set layout.");
        }

        pressure_kernel_descriptor_set_ = pressure_kernel_descriptor_set_layout_->allocate(descriptor_pool_->get());
    }

    // Velocity update
    {
        velocity_update_set_layout_ = lava::descriptor::make();
        velocity_update_set_layout_->add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                 VK_SHADER_STAGE_COMPUTE_BIT); // Pressure field A
        velocity_update_set_layout_->add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                 VK_SHADER_STAGE_COMPUTE_BIT); // Advected velocity field
        velocity_update_set_layout_->add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                 VK_SHADER_STAGE_COMPUTE_BIT); // Fixed velocity field

        if (!velocity_update_set_layout_->create(app_.device))
        {
            lava::logger()->error("Failed to create velocity descriptor set layout.");
            throw std::runtime_error("Failed to create velocity descriptor set layout.");
        }

        velocity_update_descriptor_set_ = velocity_update_set_layout_->allocate(descriptor_pool_->get());
    }

    // Advect color
    {
        advect_color_set_layout_ = lava::descriptor::make();
        advect_color_set_layout_->add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                              VK_SHADER_STAGE_COMPUTE_BIT); // Velocity field
        advect_color_set_layout_->add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                              VK_SHADER_STAGE_COMPUTE_BIT); // Previous color field
        advect_color_set_layout_->add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                              VK_SHADER_STAGE_COMPUTE_BIT); // Advected color field

        if (!advect_color_set_layout_->create(app_.device))
        {
            lava::logger()->error("Failed to create advect color descriptor set layout.");
            throw std::runtime_error("Failed to create advect color descriptor set layout.");
        }

        advect_color_descriptor_set_ = advect_color_set_layout_->allocate(descriptor_pool_->get());
    }

    // Color update
    {
        color_update_set_layout_ = lava::descriptor::make();
        color_update_set_layout_->add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                              VK_SHADER_STAGE_COMPUTE_BIT); // Temporary color field
        color_update_set_layout_->add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                              VK_SHADER_STAGE_COMPUTE_BIT); // Target color field

        if (!color_update_set_layout_->create(app_.device))
        {
            lava::logger()->error("Failed to create color update descriptor set layout.");
            throw std::runtime_error("Failed to create color update descriptor set layout.");
        }

        color_update_descriptor_set_ = color_update_set_layout_->allocate(descriptor_pool_->get());
    }
}

void Simulation::SetupPipelines()
{
    auto create_pipeline = [&](lava::compute_pipeline::s_ptr &pipeline, const char *shader_name,
                               lava::descriptor::s_ptr descriptor_set_layout, VkShaderStageFlagBits stage,
                               lava::pipeline_layout::s_ptr &existing_pipeline_layout)
    {
        pipeline = lava::compute_pipeline::make(app_.device);

        lava::c_data shader_data = app_.producer.get_shader(shader_name);
        if (!shader_data.addr)
        {
            throw std::runtime_error(std::string("Failed to load shader: ") + shader_name);
        }

        if (!existing_pipeline_layout)
        {
            existing_pipeline_layout = lava::pipeline_layout::make();
            existing_pipeline_layout->add(descriptor_set_layout);
            existing_pipeline_layout->add_push_constant_range(
                {static_cast<unsigned int>(stage), 0, sizeof(SimulationConstants)});

            if (!existing_pipeline_layout->create(app_.device))
            {
                throw std::runtime_error(std::string("Failed to create pipeline layout for: ") + shader_name);
            }
        }

        pipeline->set_shader_stage(shader_data, stage);
        pipeline->set_layout(existing_pipeline_layout);

        if (!pipeline->create())
        {
            throw std::runtime_error(std::string("Failed to create pipeline for: ") + shader_name);
        }
    };

    create_pipeline(advect_pipeline_, "VelocityAdvection.comp", advect_descriptor_set_layout_,
                    VK_SHADER_STAGE_COMPUTE_BIT, advect_pipeline_layout_);

    create_pipeline(divergence_pipeline_, "DivergenceCalculation.comp", divergence_descriptor_set_layout_,
                    VK_SHADER_STAGE_COMPUTE_BIT, divergence_pipeline_layout_);

    create_pipeline(pressure_jacobi_pipeline_, "PressureProjectionJacobi.comp", pressure_jacobi_descriptor_set_layout_,
                    VK_SHADER_STAGE_COMPUTE_BIT, pressure_jacobi_pipeline_layout_);

    create_pipeline(pressure_kernel_pipeline_, "PressureProjectionKernel.comp", pressure_kernel_descriptor_set_layout_,
                    VK_SHADER_STAGE_COMPUTE_BIT, pressure_kernel_pipeline_layout_);

    create_pipeline(velocity_update_pipeline_, "VelocityUpdate.comp", velocity_update_set_layout_,
                    VK_SHADER_STAGE_COMPUTE_BIT, velocity_update_pipeline_layout_);

    create_pipeline(advect_color_pipeline_, "ColorAdvection.comp", advect_color_set_layout_,
                    VK_SHADER_STAGE_COMPUTE_BIT, advect_color_pipeline_layout_);

    create_pipeline(color_update_pipeline_, "ColorUpdate.comp", color_update_set_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                    color_update_pipeline_layout_);
}

void Simulation::UpdateDescriptorSets()
{
    auto write_descriptor_sets = [&](VkDescriptorSet descriptor_set,
                                     const std::vector<VkDescriptorImageInfo> &image_infos,
                                     const std::vector<VkDescriptorType> &descriptor_types)
    {
        std::vector<VkWriteDescriptorSet> write_sets;
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
    };

    // Velocity advection
    {
        VkDescriptorImageInfo velocity_field_info = {.sampler = velocity_field_texture_->get_sampler(),
                                                     .imageView = velocity_field_texture_->get_image()->get_view(),
                                                     .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo advected_velocity_field_info = {
            .sampler = VK_NULL_HANDLE,
            .imageView = advected_velocity_field_texture_->get_image()->get_view(),
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

        write_descriptor_sets(advect_descriptor_set_, {velocity_field_info, advected_velocity_field_info},
                              {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE});
    }

    // Divergence calculation
    {
        VkDescriptorImageInfo advected_velocity_field_info = {
            .sampler = VK_NULL_HANDLE,
            .imageView = advected_velocity_field_texture_->get_image()->get_view(),
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorImageInfo divergence_field_info = {.sampler = VK_NULL_HANDLE,
                                                       .imageView = divergence_field_texture_->get_image()->get_view(),
                                                       .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

        write_descriptor_sets(divergence_descriptor_set_, {advected_velocity_field_info, divergence_field_info},
                              {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE});
    }

    // Pressure projection using Jacobi iteration
    {
        VkDescriptorImageInfo divergence_field_info = {.sampler = VK_NULL_HANDLE,
                                                       .imageView = divergence_field_texture_->get_image()->get_view(),
                                                       .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorImageInfo pressure_field_A_info = {.sampler = VK_NULL_HANDLE,
                                                       .imageView = pressure_field_jacobi_texture_A_->get_image()->get_view(),
                                                       .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorImageInfo pressure_field_B_info = {.sampler = VK_NULL_HANDLE,
                                                       .imageView = pressure_field_jacobi_texture_B_->get_image()->get_view(),
                                                       .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

        write_descriptor_sets(
            pressure_jacobi_descriptor_set_A_, {divergence_field_info, pressure_field_A_info, pressure_field_B_info},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE});
        write_descriptor_sets(
            pressure_jacobi_descriptor_set_B_, {divergence_field_info, pressure_field_B_info, pressure_field_A_info},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE});
    }

    // Pressure projection using unified kernel
    {
        VkDescriptorImageInfo divergence_field_info = {.sampler = VK_NULL_HANDLE,
                                                       .imageView = divergence_field_texture_->get_image()->get_view(),
                                                       .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

        // Share the same pressure field texture with Jacobi method
        VkDescriptorImageInfo pressure_field_info = {.sampler = VK_NULL_HANDLE,
                                                       .imageView =
                                                         pressure_field_jacobi_texture_A_->get_image()->get_view(),
                                                       .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

        write_descriptor_sets(
            pressure_kernel_descriptor_set_, {divergence_field_info, pressure_field_info},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE});
    }

    // Velocity update
    {
        VkDescriptorImageInfo pressure_field_A_info = {.sampler = VK_NULL_HANDLE,
                                                       .imageView = pressure_field_jacobi_texture_A_->get_image()->get_view(),
                                                       .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorImageInfo advected_velocity_field_info = {
            .sampler = VK_NULL_HANDLE,
            .imageView = advected_velocity_field_texture_->get_image()->get_view(),
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorImageInfo velocity_field_info = {.sampler = VK_NULL_HANDLE,
                                                     .imageView = velocity_field_texture_->get_image()->get_view(),
                                                     .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

        write_descriptor_sets(
            velocity_update_descriptor_set_, {pressure_field_A_info, advected_velocity_field_info, velocity_field_info},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE});
    }

    // Advect color
    {
        VkDescriptorImageInfo velocity_field_info = {.sampler = velocity_field_texture_->get_sampler(),
                                                     .imageView = velocity_field_texture_->get_image()->get_view(),
                                                     .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo color_field_A_info = {.sampler = color_field_texture_A_->get_sampler(),
                                                    .imageView = color_field_texture_A_->get_image()->get_view(),
                                                    .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo color_field_B_info = {.sampler = VK_NULL_HANDLE,
                                                    .imageView = color_field_texture_B_->get_image()->get_view(),
                                                    .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

        write_descriptor_sets(advect_color_descriptor_set_,
                              {velocity_field_info, color_field_A_info, color_field_B_info},
                              {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                               VK_DESCRIPTOR_TYPE_STORAGE_IMAGE});
    }

    // Update color field
    {
        VkDescriptorImageInfo color_field_A_info = {.sampler = color_field_texture_A_->get_sampler(),
                                                    .imageView = color_field_texture_A_->get_image()->get_view(),
                                                    .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorImageInfo color_field_B_info = {.sampler = color_field_texture_B_->get_sampler(),
                                                    .imageView = color_field_texture_B_->get_image()->get_view(),
                                                    .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

        write_descriptor_sets(color_update_descriptor_set_, {color_field_B_info, color_field_A_info},
                              {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE});
    }
}

void Simulation::JacobiPressureProjection(VkCommandBuffer cmd_buffer, const SimulationConstants &constants)
{
    divergence_field_texture_->get_image()->transition_layout(
        cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    pressure_jacobi_pipeline_->bind(cmd_buffer);

    vkCmdPushConstants(cmd_buffer, pressure_jacobi_pipeline_->get_layout()->get(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(SimulationConstants), &constants);

    for (uint32_t i = 0; i < pressure_jacobi_iterations_; i++)
    {
        uint32_t phase = i % 2;

        pressure_field_jacobi_texture_A_->get_image()->transition_layout(
            cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, phase ? VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        pressure_field_jacobi_texture_B_->get_image()->transition_layout(
            cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, phase ? VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        VkDescriptorSet pressure_descriptor_set = phase ? pressure_jacobi_descriptor_set_B_ : pressure_jacobi_descriptor_set_A_;

        vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pressure_jacobi_pipeline_->get_layout()->get(), 0,
                                1, &pressure_descriptor_set, 0, nullptr);

        vkCmdDispatch(cmd_buffer, group_count_x_, group_count_y_, 1);
    }

    // Swap from textures by handles since the final result is always stored in texture A
    if (pressure_jacobi_iterations_ % 2 != 0)
    {
        std::swap(pressure_field_jacobi_texture_A_, pressure_field_jacobi_texture_B_);
    }
}

void Simulation::KernelPressureProjection(VkCommandBuffer cmd_buffer, const SimulationConstants &constants)
{
    divergence_field_texture_->get_image()->transition_layout(
        cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    pressure_field_jacobi_texture_A_->get_image()->transition_layout(
        cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    pressure_kernel_pipeline_->bind(cmd_buffer);

    vkCmdPushConstants(cmd_buffer, pressure_kernel_pipeline_->get_layout()->get(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(SimulationConstants), &constants);

    vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pressure_kernel_pipeline_->get_layout()->get(),
                            0, 1, &pressure_kernel_descriptor_set_, 0, nullptr);

    vkCmdDispatch(cmd_buffer, group_count_x_, group_count_y_, 1);


}

void Simulation::OnUpdate(VkCommandBuffer cmd_buffer, const FrameTimeInfo &frame_context)
{
    const lava::uv2 window_size = app_.target->get_size();
    float delta_time = glm::clamp(frame_context.delta_time, 0.0f, 1.0f / 30.0f);

    SimulationConstants constants{};
    constants.current_time = static_cast<float>(frame_context.current_time);
    constants.delta_time = delta_time;
    constants.texture_width = static_cast<int>(window_size.x);
    constants.texture_height = static_cast<int>(window_size.y);
    constants.fluid_density = 0.5f;
    constants.vorticity_strength = 0.5f;
    constants.reset_color = reset_flag_;

    reset_flag_ = false;

    // Advect velocity pass
    {
        velocity_field_texture_->get_image()->transition_layout(cmd_buffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                                VK_ACCESS_SHADER_READ_BIT,
                                                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        advected_velocity_field_texture_->get_image()->transition_layout(
            cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        advect_pipeline_->bind(cmd_buffer);

        vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, advect_pipeline_->get_layout()->get(), 0, 1,
                                &advect_descriptor_set_, 0, nullptr);

        vkCmdPushConstants(cmd_buffer, advect_pipeline_->get_layout()->get(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(SimulationConstants), &constants);

        vkCmdDispatch(cmd_buffer, group_count_x_, group_count_y_, 1);
    }

    // Calculate divergence pass
    {
        advected_velocity_field_texture_->get_image()->transition_layout(
            cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        divergence_field_texture_->get_image()->transition_layout(
            cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        divergence_pipeline_->bind(cmd_buffer);

        vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, divergence_pipeline_->get_layout()->get(),
                                0, 1, &divergence_descriptor_set_, 0, nullptr);

        vkCmdPushConstants(cmd_buffer, divergence_pipeline_->get_layout()->get(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(SimulationConstants), &constants);

        vkCmdDispatch(cmd_buffer, group_count_x_, group_count_y_, 1);
    }

    // Project pressure passes
    {
        if (pressure_projection_method_ == PressureProjectionMethod::Jacobi)
        {
            JacobiPressureProjection(cmd_buffer, constants);
        }
        else if (pressure_projection_method_ == PressureProjectionMethod::Kernel)
        {
            KernelPressureProjection(cmd_buffer, constants);
        }
    }

    // Update velocity pass
    {
        pressure_field_jacobi_texture_A_->get_image()->transition_layout(
            cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        velocity_field_texture_->get_image()->transition_layout(
            cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        velocity_update_pipeline_->bind(cmd_buffer);

        vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                velocity_update_pipeline_->get_layout()->get(), 0, 1, &velocity_update_descriptor_set_,
                                0, nullptr);

        vkCmdPushConstants(cmd_buffer, velocity_update_pipeline_->get_layout()->get(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(SimulationConstants), &constants);

        vkCmdDispatch(cmd_buffer, group_count_x_, group_count_y_, 1);
    }

    // Advect color pass
    {
        velocity_field_texture_->get_image()->transition_layout(cmd_buffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                                VK_ACCESS_SHADER_READ_BIT,
                                                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        color_field_texture_A_->get_image()->transition_layout(cmd_buffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                               VK_ACCESS_SHADER_READ_BIT,
                                                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        color_field_texture_B_->get_image()->transition_layout(
            cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        advect_color_pipeline_->bind(cmd_buffer);

        vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, advect_color_pipeline_->get_layout()->get(),
                                0, 1, &advect_color_descriptor_set_, 0, nullptr);

        vkCmdPushConstants(cmd_buffer, advect_color_pipeline_->get_layout()->get(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(SimulationConstants), &constants);

        vkCmdDispatch(cmd_buffer, group_count_x_, group_count_y_, 1);
    }

    // Update color texture
    {
        color_field_texture_A_->get_image()->transition_layout(
            cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        color_field_texture_B_->get_image()->transition_layout(
            cmd_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        color_update_pipeline_->bind(cmd_buffer);

        vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, color_update_pipeline_->get_layout()->get(),
                                0, 1, &color_update_descriptor_set_, 0, nullptr);

        vkCmdPushConstants(cmd_buffer, color_update_pipeline_->get_layout()->get(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(SimulationConstants), &constants);

        vkCmdDispatch(cmd_buffer, group_count_x_, group_count_y_, 1);
    }

    // Transition resources for rendering
    {
        pressure_field_jacobi_texture_A_->get_image()->transition_layout(cmd_buffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                                  VK_ACCESS_SHADER_READ_BIT,
                                                                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        divergence_field_texture_->get_image()->transition_layout(cmd_buffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                                  VK_ACCESS_SHADER_READ_BIT,
                                                                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        velocity_field_texture_->get_image()->transition_layout(cmd_buffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                                VK_ACCESS_SHADER_READ_BIT,
                                                                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        color_field_texture_A_->get_image()->transition_layout(cmd_buffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                               VK_ACCESS_SHADER_READ_BIT,
                                                               VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    }
}

}