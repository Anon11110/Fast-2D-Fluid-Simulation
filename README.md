
# Fast 2D Fluid Simulation

## Overview

This project is a 2D incompressible Euler grid fluid simulation implemented using Vulkan. The simulation focuses on solving the Navier-Stokes equations for incompressible fluids, with a particular emphasis on the pressure projection step.

## Key Features

- **Jacobi Iteration for Pressure Calculation**: Implements the classic iterative solver for the pressure projection step.
- **Poisson Filter-Based Pressure Calculation**: Utilizes the compact Poisson filters described in the paper *["Compact Poisson Filters for Fast Fluid Simulation, Siggraph 2022"](https://doi.org/10.1145/3528233.3530737)* for efficient, high-performance pressure solving.
- **V-Cycle Multigrid Solver**:
  - Combines coarse and fine grid resolution to accelerate convergence.
  - Supports both **Jacobi iteration** and **Poisson filter** as smoothers.

## Dependencies

- [**liblava**](https://github.com/liblava/liblava): Vulkan abstraction framework, but still maintains low-level control.
  
## Methods Implemented

1. **Jacobi Iteration**:
   - Classic iterative method for solving the pressure projection equation.

2. **Poisson Filter-Based Solver**:
   - Efficient, direct solver based on compact Poisson filters.
   - Approximate multiple Jacobi iterations with one convolution.

3. **V-Cycle Multigrid Solver**:
   - Combines coarse and fine grid levels to improve convergence.
   - Two options for smoothers:
     - **Jacobi Iteration**: Traditional smoother for multigrid solvers.
     - **Poisson Filter**: Faster smoother derived from compact Poisson filters.

     
4. **Semi-Lagrangian Advection for Velocity**:
   - Implements the semi-Lagrangian method for velocity advection.
   - Computes new velocities by tracing particle paths backward in time and interpolating values from previous positions.
