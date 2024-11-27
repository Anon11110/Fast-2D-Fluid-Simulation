#include "Simulation.hpp"

namespace FluidSimulation
{
Simulation::Simulation(lava::engine &app) : app_(app)
{
    const lava::uv2 window_size = app_.target->get_size();

    AddShaderMappings();
    CreateTextures();
    CreateBuffers();
    CreateDescriptorPool();
    CreateComputePasses();
}

Simulation::~Simulation()
{
    if (descriptor_pool_)
        descriptor_pool_->destroy();

    FluidSimulation::ResourceManager::GetInstance(&app_).DestroyAllResources();
}

void Simulation::AddShaderMappings()
{
    const std::vector<std::pair<std::string, std::string>> file_mappings{
        {"ObstacleMaskFilling.comp", "../shaders/ObstacleMaskFilling.comp"},

        {"VelocityAdvection.comp", "../shaders/VelocityAdvection.comp"},

        {"DivergenceCalculation.comp", "../shaders/DivergenceCalculation.comp"},

        {"PressureProjectionJacobi.comp", "../shaders/PressureProjectionJacobi.comp"},

        {"PressureProjectionKernel.comp", "../shaders/PressureProjectionKernel.comp"},

        {"VelocityUpdate.comp", "../shaders/VelocityUpdate.comp"},

        {"ColorAdvection.comp", "../shaders/ColorAdvection.comp"},

        {"ColorUpdate.comp", "../shaders/ColorUpdate.comp"},

        {"PressureRelaxation.comp", "../shaders/PressureRelaxation.comp"},

        {"PressureRestriction.comp", "../shaders/PressureRestriction.comp"},

        {"PressureProlongation.comp", "../shaders/PressureProlongation.comp"},

        {"PressureRelaxationPoisson.comp", "../shaders/PressureRelaxationPoisson.comp"},

        {"ResidualErrorCalculation.comp", "../shaders/ResidualErrorCalculation.comp"},

        {"ResidualReduction.comp", "../shaders/ResidualReduction.comp"}};

    for (auto &&[name, file] : file_mappings)
    {
        app_.props.add(name, file);
    }
}

void Simulation::CreateMultigridTextures(uint32_t max_levels)
{
    auto &resource_manager = FluidSimulation::ResourceManager::GetInstance(&app_);

    for (uint32_t level = 0; level < max_levels; level++)
    {
        glm::uvec2 texture_size{std::max(1u, app_.target->get_size().x / (1 << level)),
                                std::max(1u, app_.target->get_size().y / (1 << level))};

        std::string suffix = "_L" + std::to_string(level);

        if (level > 0)
        {
            resource_manager.CreateTexture(
                "pressure_multigrid_A" + suffix,
                {texture_size, VK_FORMAT_R16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                 VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR});

            resource_manager.CreateTexture(
                "pressure_multigrid_B" + suffix,
                {texture_size, VK_FORMAT_R16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                 VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR});
        }

        resource_manager.CreateTexture(
            "multigrid_temp" + suffix,
            {texture_size, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
             VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR});

        resource_manager.CreateTexture(
            "multigrid_temp1" + suffix,
            {texture_size, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
             VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR});
    }
}

void Simulation::CreateTextures()
{
    const glm::uvec2 window_size = app_.target->get_size();

    auto &resource_manager = FluidSimulation::ResourceManager::GetInstance(&app_);

    auto create_resource_texture = [&](const std::string &name, VkFormat format, VkImageUsageFlags usage,
                                       VkSamplerAddressMode address_mode, VkFilter filter,
                                       VkSamplerMipmapMode mipmap_mode, glm::uvec2 size)
    {
        FluidSimulation::TextureCreateInfo create_info = {size, format, usage, address_mode, filter, mipmap_mode};
        resource_manager.CreateTexture(name, create_info);
    };

    create_resource_texture(
        "obstacle_mask", VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST, window_size);

    create_resource_texture(
        "velocity_field", VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, window_size);

    create_resource_texture(
        "advected_velocity_field", VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, window_size);

    create_resource_texture(
        "divergence_field", VK_FORMAT_R16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, window_size);

    create_resource_texture(
        "pressure_field_A", VK_FORMAT_R16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, window_size);

    create_resource_texture(
        "pressure_field_B", VK_FORMAT_R16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, window_size);

    create_resource_texture(
        "color_field_A", VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, window_size);

    create_resource_texture(
        "color_field_B", VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, window_size);

    create_resource_texture(
        "temp", VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, window_size);

    create_resource_texture(
        "temp1", VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, window_size);

    create_resource_texture("residual", VK_FORMAT_R32_SFLOAT,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                            VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST,
                            window_size);

    CreateMultigridTextures(multigrid_levels_);
}

void Simulation::CreateBuffers()
{
    const lava::uv2 texture_size = app_.target->get_size();

    auto &resource_manager = FluidSimulation::ResourceManager::GetInstance(&app_);
    resource_manager.CreateBuffer("staging_buffer", texture_size.x * texture_size.y * sizeof(float),
                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
}

void Simulation::CreateDescriptorPool()
{
    descriptor_pool_ = lava::descriptor::pool::make();
    descriptor_pool_->create(
        app_.device, {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 100}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 100}}, 100);
}

void Simulation::CreateComputePasses()
{
    obstacle_filling_pass_ = ObstacleFillingPass::Make(app_, descriptor_pool_);
    obstacle_filling_pass_->SetNeedsUpdate(upload_obstacle_mask_);

    velocity_advect_pass_ = VelocityAdvectionPass::Make(app_, descriptor_pool_);

    divergence_calculation_pass_ = DivergenceCalculationPass::Make(app_, descriptor_pool_);

    jacobi_pressure_projection_pass_ = JacobiPressurePass::Make(app_, descriptor_pool_);

    poisson_pressure_projection_pass_ = PoissonPressurePass::Make(app_, descriptor_pool_);

    v_cycle_pressure_projection_pass_ = VCyclePressurePass::Make(app_, descriptor_pool_);
    v_cycle_pressure_projection_pass_->SetMaxLevels(multigrid_levels_);
    v_cycle_pressure_projection_pass_->SetRelaxationIterations(relaxation_iterations_);

    velocity_update_pass_ = VelocityUpdatePass::Make(app_, descriptor_pool_);

    color_advect_pass_ = ColorAdvectPass::Make(app_, descriptor_pool_);

    color_update_pass_ = ColorUpdatePass::Make(app_, descriptor_pool_);

    residual_calculation_pass_ = ResidualCalculationPass::Make(app_, descriptor_pool_);
}

void Simulation::OnUpdate(VkCommandBuffer cmd_buffer, const FrameTimeInfo &frame_context)
{
    const lava::uv2 window_size = app_.target->get_size();
    float delta_time = glm::clamp(frame_context.delta_time, 0.0f, 1.0f / 30.0f);

    SimulationConstants simulation_constants{};
    simulation_constants.current_time = static_cast<float>(frame_context.current_time);
    simulation_constants.delta_time = delta_time;
    simulation_constants.texture_width = static_cast<int>(window_size.x);
    simulation_constants.texture_height = static_cast<int>(window_size.y);
    simulation_constants.divergence_width = static_cast<int>(window_size.x);
    simulation_constants.divergence_height = static_cast<int>(window_size.y);
    simulation_constants.fluid_density = 0.5f;
    simulation_constants.vorticity_strength = 0.5f;
    simulation_constants.reset_color = static_cast<int>(reset_flag_);

    reset_flag_ = false;

    MultigridConstants multigrid_constants;
    multigrid_constants.fine_width = app_.target->get_size().x / (1 << multigrid_levels_);
    multigrid_constants.fine_height = app_.target->get_size().y / (1 << multigrid_levels_);
    multigrid_constants.coarse_width = app_.target->get_size().x / (1 << (multigrid_levels_ + 1));
    multigrid_constants.coarse_height = app_.target->get_size().y / (1 << (multigrid_levels_ + 1));

    obstacle_filling_pass_->Execute(cmd_buffer, simulation_constants);

    velocity_advect_pass_->Execute(cmd_buffer, simulation_constants);

    divergence_calculation_pass_->Execute(cmd_buffer, simulation_constants);

    if (pressure_projection_method_ == PressureProjectionMethod::Jacobi)
    {
        jacobi_pressure_projection_pass_->Execute(cmd_buffer, simulation_constants);
    }
    else if (pressure_projection_method_ == PressureProjectionMethod::Poisson_Filter)
    {
        poisson_pressure_projection_pass_->Execute(cmd_buffer, simulation_constants);
    }
    else if (pressure_projection_method_ == PressureProjectionMethod::Multigrid ||
             pressure_projection_method_ == PressureProjectionMethod::Multigrid_Poisson)
    {
        VCycleRelaxationType relaxation_type = (pressure_projection_method_ == PressureProjectionMethod::Multigrid)
                                                   ? VCycleRelaxationType::Standard
                                                   : VCycleRelaxationType::Poisson_Filter;

        v_cycle_pressure_projection_pass_->SetRelaxationType(relaxation_type);
        v_cycle_pressure_projection_pass_->Execute(cmd_buffer, simulation_constants);
    }

    if (calculate_residual_error_ && frame_count_ == PRESSURE_CONVERGENCE_CHECK_FRAME)
    {
        residual_calculation_pass_->Execute(cmd_buffer, simulation_constants);
        auto staging_buffer = FluidSimulation::ResourceManager::GetInstance(&app_).GetBuffer("staging_buffer");
        residual_calculation_pass_->CopyResidualToCPU(cmd_buffer, staging_buffer);
    }

    velocity_update_pass_->Execute(cmd_buffer, simulation_constants);

    color_advect_pass_->Execute(cmd_buffer, simulation_constants);

    color_update_pass_->Execute(cmd_buffer, simulation_constants);

    // Transition resources for rendering
    auto &resource_manager = ResourceManager::GetInstance();

    auto color_field_texture = resource_manager.GetTexture("color_field_A");
    color_field_texture->get_image()->transition_layout(cmd_buffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                        VK_ACCESS_SHADER_READ_BIT,
                                                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    auto obstacle_mask_ = resource_manager.GetTexture("obstacle_mask");
    obstacle_mask_->get_image()->transition_layout(cmd_buffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                   VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    frame_count_++;
}

} // namespace FluidSimulation