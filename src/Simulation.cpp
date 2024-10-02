#include "Simulation.hpp"

namespace FluidSimulation
{
Simulation::Simulation(lava::engine &app) : app_(app)
{
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
}

void Simulation::AddShaderMappings()
{
    const std::vector<std::pair<std::string, std::string>> file_mappings{
        {"VelocityAdvection.comp", "../shaders/VelocityAdvection.comp"},
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

    uint32_t group_count_x = (window_size.x - 1) / 16 + 1;
    uint32_t group_count_y = (window_size.y - 1) / 16 + 1;

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

        vkCmdDispatch(cmd_buffer, group_count_x, group_count_y, 1);
    }
}

}