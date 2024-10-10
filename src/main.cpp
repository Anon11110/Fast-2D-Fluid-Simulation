#include "FluidRenderer.hpp"
#include "Simulation.hpp"
#include "imgui.h"
#include "liblava/lava.hpp"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

using namespace lava;

int run(int argc, char *argv[])
{
    frame_env env;
    env.info.app_name = "VkFluidSimulation";
    env.cmd_line = {argc, argv};
    env.info.req_api_version = api_version::v1_3;

    engine app(env);
    if (!app.setup())
        return error::not_ready;

    FluidSimulation::FluidRenderer::s_ptr fluid_renderer = FluidSimulation::FluidRenderer::make(app);
    auto render_pipeline = fluid_renderer->GetPipeline();

    target_callback swapchain_callback;
    swapchain_callback.on_created = [&](VkAttachmentsRef, rect::ref)
    {
        if (fluid_renderer)
        {
            fluid_renderer->Destroy();
        }

        fluid_renderer = FluidSimulation::FluidRenderer::make(app);

        render_pass::s_ptr render_pass = app.shading.get_pass();
        render_pass->remove(render_pipeline);
        auto new_pipeline = fluid_renderer->GetPipeline();
        render_pass->add_front(new_pipeline);
        render_pipeline = new_pipeline;

        return true;
    };
    swapchain_callback.on_destroyed = [&]() {};

    app.on_create = [&]()
    {
        render_pass::s_ptr render_pass = app.shading.get_pass();
        app.target->add_callback(&swapchain_callback);
        render_pass->add_front(render_pipeline);
        return true;
    };

    app.on_process = [&](VkCommandBuffer cmd_buffer, uint32_t frame)
    {
        double current_time = glfwGetTime();
        float delta_time = static_cast<float>(current_time - fluid_renderer->GetLastFrameTime());
        fluid_renderer->SetLastFrameTime(current_time);

        FluidSimulation::FrameTimeInfo frame_context{current_time, delta_time};
        fluid_renderer->OnCompute(cmd_buffer, frame_context);
    };

    app.on_destroy = [&]() {

    };

    bool reload = false;
    app.input.key.listeners.add(
        [&](key_event::ref event)
        {
            if (app.imgui.capture_mouse())
                return false;

            if (event.pressed(key::enter, mod::control))
            {
                reload = true;
                app.shut_down();
                return input_done;
            }

            if (event.pressed(key::escape))
            {
                app.shut_down();
                return input_done;
            }

            return false;
        });

    app.imgui.layers.add("info",
                         [&]()
                         {
                             ImGui::SetNextWindowPos({30, 30}, ImGuiCond_FirstUseEver);
                             ImGui::SetNextWindowSize({260, 135}, ImGuiCond_FirstUseEver);

                             ImGui::Begin(app.get_name());

                             uv2 target_size = app.target->get_size();
                             ImGui::Text("target: %d x %d", target_size.x, target_size.y);

                             ImGui::SameLine();

                             ImGui::Text("frames: %d", app.target->get_frame_count());

                             app.draw_about();

                             ImGui::End();
                         });

    app.add_run_end([&]() {});

    auto result = app.run();
    if (result != 0)
        return result;

    return reload ? 114514 : 0;
}

int main(int argc, char *argv[])
{
    int ret;
    do
    {
        ret = run(argc, argv);
    } while (ret == 114514);
    return ret;
}
