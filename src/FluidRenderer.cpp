#include "FluidRenderer.hpp"

namespace FluidSimulation
{
FluidRenderer::FluidRenderer(lava::engine &app) : app_(app)
{
    swapchain_images_count_ = app_.target->get_frame_count();

    FluidSimulation::ResourceManager::GetInstance(&app).Initialize(&app);
    simulation_ = Simulation::Make(app);
    AddShaderMappings();
    CreateDescriptorPool();
    CreateDescriptorSets();
    UpdateDescriptorSets();
    CreatePipeline();
}

FluidRenderer::~FluidRenderer()
{
    Destroy();
}

void FluidRenderer::Destroy()
{
    if (render_descriptor_set_layout_)
    {
        render_descriptor_set_layout_->destroy();
        render_descriptor_set_layout_.reset();
    }

    if (descriptor_pool_)
    {
        descriptor_pool_->destroy();
        descriptor_pool_.reset();
    }

    if (render_pipeline_layout_)
    {
        render_pipeline_layout_->destroy();
        render_pipeline_layout_.reset();
    }

    if (render_pipeline_)
    {
        render_pipeline_->destroy();
        render_pipeline_.reset();
    }

    lava::logger()->debug("Fluid renderer resources destroyed.");
}

void FluidRenderer::OnCompute(VkCommandBuffer cmd_buffer, const FrameTimeInfo &frame_context)
{
    simulation_->OnUpdate(cmd_buffer, frame_context);
}

void FluidRenderer::AddShaderMappings()
{
    std::vector<std::pair<std::string, std::string>> file_mappings{
        {"Blit.frag", "../shaders/Blit.frag"},

        {"Blit.vert", "../shaders/Blit.vert"},
    };

    for (auto &&[name, file] : file_mappings)
    {
        app_.props.add(name, file);
    }
}

void FluidRenderer::CreateDescriptorPool()
{
    descriptor_pool_ = lava::descriptor::pool::make();

    constexpr uint32_t set_count = 1;

    std::vector<VkDescriptorPoolSize> pool_sizes = {
        VkDescriptorPoolSize{.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 20},
        VkDescriptorPoolSize{.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 10},
        VkDescriptorPoolSize{.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 10}};

    if (!descriptor_pool_->create(app_.device, pool_sizes, set_count, 0))
    {
        lava::logger()->error("Failed to create descriptor pool.");
        throw std::runtime_error("Failed to create descriptor pool.");
    }
}

void FluidRenderer::CreateDescriptorSets()
{
    render_descriptor_set_layout_ = lava::descriptor::make();

    render_descriptor_set_layout_->add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                               VK_SHADER_STAGE_FRAGMENT_BIT); // Color texture
    render_descriptor_set_layout_->add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                               VK_SHADER_STAGE_FRAGMENT_BIT); // Obstacle mask

    if (!render_descriptor_set_layout_->create(app_.device))
    {
        lava::logger()->error("Failed to create render descriptor set layout.");
        throw std::runtime_error("Failed to create render descriptor set layout.");
    }

    render_descriptor_set_ = render_descriptor_set_layout_->allocate(descriptor_pool_->get());

    if (!render_descriptor_set_)
    {
        lava::logger()->error("Failed to allocate render descriptor set.");
        throw std::runtime_error("Failed to allocate render descriptor set.");
    }
}

void FluidRenderer::UpdateDescriptorSets()
{
    auto &resource_manager = ResourceManager::GetInstance();

    auto color_texture = resource_manager.GetTexture("color_field_A");
    VkDescriptorImageInfo color_texture_info = {.sampler = color_texture->get_sampler(),
                                                .imageView = color_texture->get_image()->get_view(),
                                                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    auto obstacle_mask = resource_manager.GetTexture("obstacle_mask");
    VkDescriptorImageInfo obstacle_mask_info = {.sampler = obstacle_mask->get_sampler(),
                                                .imageView = obstacle_mask->get_image()->get_view(),
                                                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    std::vector<VkWriteDescriptorSet> write_descriptor_sets = {
        VkWriteDescriptorSet{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             .dstSet = render_descriptor_set_,
                             .dstBinding = 0,
                             .descriptorCount = 1,
                             .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                             .pImageInfo = &color_texture_info},
        VkWriteDescriptorSet{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             .dstSet = render_descriptor_set_,
                             .dstBinding = 1,
                             .descriptorCount = 1,
                             .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                             .pImageInfo = &obstacle_mask_info}};

    app_.device->vkUpdateDescriptorSets(uint32_t(write_descriptor_sets.size()), write_descriptor_sets.data(), 0,
                                        nullptr);
}

void FluidRenderer::CreatePipeline()
{
    auto render_pass = app_.shading.get_pass();

    render_pipeline_layout_ = lava::pipeline_layout::make();
    render_pipeline_layout_->add(render_descriptor_set_layout_);
    if (!render_pipeline_layout_->create(app_.device))
    {
        lava::logger()->error("Failed to create render pipeline layout.");
        throw std::runtime_error("Failed to create render pipeline layout.");
    }

    render_pipeline_ = lava::render_pipeline::make(app_.device, app_.pipeline_cache);

    if (!render_pipeline_->add_shader(app_.producer.get_shader("Blit.vert"), VK_SHADER_STAGE_VERTEX_BIT))
    {
        lava::logger()->error("Failed to load fluid rendering vertex shader.");
        throw std::runtime_error("Failed to load fluid rendering vertex shader.");
    }

    if (!render_pipeline_->add_shader(app_.producer.get_shader("Blit.frag"), VK_SHADER_STAGE_FRAGMENT_BIT))
    {
        lava::logger()->error("Failed to load fluid rendering fragment shader.");
        throw std::runtime_error("Failed to load fluid rendering fragment shader.");
    }

    render_pipeline_->set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    render_pipeline_->set_depth_test_and_write(false, false);
    render_pipeline_->set_rasterization_polygon_mode(VK_POLYGON_MODE_FILL);
    render_pipeline_->set_rasterization_cull_mode(VK_CULL_MODE_NONE);
    render_pipeline_->set_rasterization_front_face(VK_FRONT_FACE_CLOCKWISE);
    render_pipeline_->add_color_blend_attachment();
    render_pipeline_->set_layout(render_pipeline_layout_);

    if (!render_pipeline_->create(render_pass->get()))
    {
        lava::logger()->error("Failed to create render pipeline.");
        throw std::runtime_error("Failed to create render pipeline.");
    }

    render_pipeline_->on_process = [&](VkCommandBuffer cmd_buffer)
    {
        render_pipeline_->bind(cmd_buffer);
        render_pipeline_->set_viewport_and_scissor(cmd_buffer, app_.target->get_size());
        render_pipeline_layout_->bind(cmd_buffer, render_descriptor_set_, 0, {}, VK_PIPELINE_BIND_POINT_GRAPHICS);

        vkCmdDraw(cmd_buffer, 3, 1, 0, 0);
    };
}

void FluidRenderer::OnRender(uint32_t frame, VkCommandBuffer cmd_buffer)
{
    lava::begin_label(cmd_buffer, "render", glm::vec4(1, 0, 0, 0));

    render_pipeline_->bind(cmd_buffer);
    render_pipeline_->set_viewport_and_scissor(cmd_buffer, app_.target->get_size());
    render_pipeline_layout_->bind(cmd_buffer, render_descriptor_set_, 0, {}, VK_PIPELINE_BIND_POINT_GRAPHICS);
    vkCmdDraw(cmd_buffer, 3, 1, 0, 0);

    lava::end_label(cmd_buffer);
}
} // namespace FluidSimulation
