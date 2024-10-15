#pragma once
#ifndef FLUID_RENDERER_HPP
#define FLUID_RENDERER_HPP

#include "Simulation.hpp"
#include <liblava/lava.hpp>

namespace FluidSimulation
{
class FluidRenderer
{
  public:
    FluidRenderer(lava::engine &app);

    ~FluidRenderer();

    void Destroy();

    void OnCompute(VkCommandBuffer cmd_buffer, const FrameTimeInfo &frame_context);
    void OnRender(uint32_t frame, VkCommandBuffer cmd_buffer);

    [[nodiscard]] lava::render_pipeline::s_ptr GetPipeline() const
    {
        return render_pipeline_;
    }

    [[nodiscard]] double GetLastFrameTime() const
    {
        return last_frame_time_;
    }

    void SetLastFrameTime(double time)
    {
        last_frame_time_ = time;
    }

    Simulation::s_ptr simulation_;

    using s_ptr = std::shared_ptr<FluidRenderer>;

    static s_ptr make(lava::engine &app)
    {
        return std::make_shared<FluidRenderer>(app);
    }

  private:
    void AddShaderMappings();
    void CreateDescriptorPool();
    void CreateDescriptorSets();
    void UpdateDescriptorSets();
    void CreatePipeline();

    lava::engine &app_;

    double last_frame_time_ = 0;

    int swapchain_images_count_ = 0;

    lava::descriptor::pool::s_ptr descriptor_pool_;

    lava::descriptor::s_ptr render_descriptor_set_layout_;
    VkDescriptorSet render_descriptor_set_{};

    lava::pipeline_layout::s_ptr render_pipeline_layout_;
    lava::render_pipeline::s_ptr render_pipeline_;
};
} // namespace FluidSimulation

#endif // FLUID_RENDERER_HPP