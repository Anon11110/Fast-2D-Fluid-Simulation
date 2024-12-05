#pragma once
#ifndef VCYCLE_PRESSURE_PASS_HPP
#define VCYCLE_PRESSURE_PASS_HPP

#include "ComputePass.hpp"
#include "ResourceManager.hpp"
#include <liblava/lava.hpp>
#include <vector>

namespace FluidSimulation
{

enum class VCycleRelaxationType
{
    Standard,
    Poisson_Filter
};

class VCyclePressurePass : public ComputePass
{
  public:
    using s_ptr = std::shared_ptr<VCyclePressurePass>;

    VCyclePressurePass(lava::engine &app, lava::descriptor::pool::s_ptr pool, uint32_t max_levels = 5);
    ~VCyclePressurePass() override;

    void CreateDescriptorSets() override;
    void UpdateDescriptorSets() override;
    void CreatePipeline() override;
    void Execute(VkCommandBuffer cmd_buffer, const SimulationConstants &constants) override;

    void SetRelaxationType(VCycleRelaxationType type)
    {
        relaxation_type_ = type;
    }
    void SetMaxLevels(uint32_t levels)
    {
        max_levels_ = levels;
    }
    void SetRelaxationIterations(uint32_t iterations)
    {
        relaxation_iterations_ = iterations;
    }
    void SetVCycleIterations(uint32_t iterations)
    {
        vcycle_iterations_ = iterations;
    }

    static s_ptr Make(lava::engine &app, lava::descriptor::pool::s_ptr pool, uint32_t max_levels = 5)
    {
        return std::make_shared<VCyclePressurePass>(app, pool, max_levels);
    }

  private:
    void CreateBasePipeline(lava::compute_pipeline::s_ptr &pipeline, const char *shader_name,
                            lava::descriptor::s_ptr descriptor_set_layout,
                            lava::pipeline_layout::s_ptr &existing_pipeline_layout, size_t push_constant_size = 0);
    void CreateRelaxationPipeline();
    void CreateResidualPipeline();
    void CreateRestrictionPipeline();
    void CreateProlongationPipeline();
    void CreatePoissonRelaxationPipeline();

    void PerformRelaxation(VkCommandBuffer cmd_buffer, const SimulationConstants &constants, uint32_t level);
    void PerformPoissonFilterRelaxation(VkCommandBuffer cmd_buffer, const SimulationConstants &constants,
                                        uint32_t level);
    void CalculateResidual(VkCommandBuffer cmd_buffer, const MultigridConstants &constants, uint32_t level);
    void PerformRestriction(VkCommandBuffer cmd_buffer, const MultigridConstants &constants, uint32_t level);
    void PerformProlongation(VkCommandBuffer cmd_buffer, const MultigridConstants &constants, uint32_t level);

    MultigridConstants CalculateMultigridConstants(uint32_t level) const;

    // Relaxation resources
    lava::descriptor::s_ptr relaxation_descriptor_set_layout_;
    std::vector<VkDescriptorSet> relaxation_descriptor_sets_A_;
    std::vector<VkDescriptorSet> relaxation_descriptor_sets_B_;
    lava::pipeline_layout::s_ptr relaxation_pipeline_layout_;
    lava::compute_pipeline::s_ptr relaxation_pipeline_;

    // Poisson relaxation resources
    lava::descriptor::s_ptr poisson_relaxation_descriptor_set_layout_;
    std::vector<VkDescriptorSet> poisson_relaxation_descriptor_sets_;
    lava::pipeline_layout::s_ptr poisson_relaxation_pipeline_layout_;
    lava::compute_pipeline::s_ptr poisson_relaxation_pipeline_;

    // Residual calculation resources
    lava::descriptor::s_ptr residual_descriptor_set_layout_;
    std::vector<VkDescriptorSet> residual_descriptor_sets_;
    lava::pipeline_layout::s_ptr residual_pipeline_layout_;
    lava::compute_pipeline::s_ptr residual_pipeline_;

    // Restriction resources
    lava::descriptor::s_ptr restriction_descriptor_set_layout_;
    std::vector<VkDescriptorSet> restriction_descriptor_sets_;
    lava::pipeline_layout::s_ptr restriction_pipeline_layout_;
    lava::compute_pipeline::s_ptr restriction_pipeline_;

    // Prolongation resources
    lava::descriptor::s_ptr prolongation_descriptor_set_layout_;
    std::vector<VkDescriptorSet> prolongation_descriptor_sets_;
    lava::pipeline_layout::s_ptr prolongation_pipeline_layout_;
    lava::compute_pipeline::s_ptr prolongation_pipeline_;

    std::vector<lava::texture::s_ptr> pressure_multigrid_texture_A_;
    std::vector<lava::texture::s_ptr> pressure_multigrid_texture_B_;
    std::vector<lava::texture::s_ptr> multigrid_temp_textures_;
    std::vector<lava::texture::s_ptr> multigrid_temp_textures1_;
    std::vector<lava::texture::s_ptr> divergence_fields_;
    lava::texture::s_ptr obstacle_mask_;

    VCycleRelaxationType relaxation_type_ = VCycleRelaxationType::Standard;
    uint32_t max_levels_ = 8;
    uint32_t relaxation_iterations_ = 2;
    uint32_t vcycle_iterations_ = 3;
};

} // namespace FluidSimulation

#endif // VCYCLE_PRESSURE_PASS_HPP