#pragma once
#ifndef FLUID_CONSTANTS_HPP
#define FLUID_CONSTANTS_HPP

#include <cstdint>

namespace FluidSimulation
{

struct FrameTimeInfo
{
    double current_time{};
    float delta_time{};
};

struct SimulationConstants
{
    float current_time;
    float delta_time;
    int texture_width;
    int texture_height;
    int divergence_width;
    int divergence_height;
    float fluid_density;
    float vorticity_strength;
    int reset_color;
};

struct MultigridConstants
{
    int fine_width;
    int fine_height;
    int coarse_width;
    int coarse_height;
};

enum class PressureProjectionMethod : uint32_t
{
    None = 0,
    Jacobi = 1 << 0,
    Poisson_Filter = 1 << 1,
    Multigrid = 1 << 2,
    Multigrid_Poisson = 1 << 3
};

inline bool HasMethod(PressureProjectionMethod method, PressureProjectionMethod flag)
{
    return (static_cast<uint32_t>(method) & static_cast<uint32_t>(flag)) != 0;
}

} // namespace FluidSimulation

#endif // FLUID_CONSTANTS_HPP