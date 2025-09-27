# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements a Julia wrapper around the mVMC (Many-Variable Variational Monte Carlo) code for quantum many-body systems. The project consists of:

- `ManyVariableVariationalMonteCarlo.jl/`: Main Julia package implementing variational Monte Carlo algorithms
- `mVMC/`: C reference implementation (git submodule from https://github.com/issp-center-dev/mVMC.git)
- `mVMC-tutorial/`: Tutorial materials and sample configurations (git submodule)

The Julia implementation mirrors the C reference while providing a more portable and testable codebase.

## Development Commands

All development should be done within the Julia package directory:

```bash
cd ManyVariableVariationalMonteCarlo.jl
```

### Package Management
```bash
# Install or update dependencies
julia --project -e 'using Pkg; Pkg.instantiate()'
```

### Testing
```bash
# Run full test suite using ReTestItems
julia --project -e 'using Pkg; Pkg.test()'

# Run specific test categories during development
julia --project -e 'using ReTestItems; ReTestItems.runtests(filter="sampler")'
```

## Code Architecture

### Module Structure
The main module `ManyVariableVariationalMonteCarlo` is organized into focused submodules:

- `src/types.jl`: Core data structures including `FaceDefinition` for config file parsing
- `src/config.jl`: Configuration file parsing and simulation setup (`SimulationConfig`)
- `src/parameters.jl`: Variational parameter management (`ParameterLayout`, `ParameterSet`)
- `src/io.jl`: File I/O operations including Green function data reading

### Key Types
- `FaceDefinition`: Represents `StdFace.def` configuration files with ordered key-value pairs
- `SimulationConfig`: Main simulation configuration parsed from input files
- `ParameterLayout`/`ParameterSet`: Manages variational parameters (RBM, Slater, OptTrans)
- `GreenFunctionTable`: Handles initial Green function data from `initial.def`

### Testing Framework
Uses ReTestItems with `@testitem` blocks instead of traditional `@test` macros. Test files must have `_tests.jl` suffix to be recognized. All stochastic tests should use `StableRNGs` for reproducibility.

## Development Guidelines

### File Organization
- Mirror the C reference implementation structure where possible
- Group related functionality by physical concepts (lattice, sampler, optimizer)
- Place temporary translation code in `src/legacy/` if immediate modernization isn't possible

### Performance Considerations
- Use preallocated buffers for hot loops
- Apply `@inbounds`/`@simd` only with safety comments
- Benchmark with `BenchmarkTools.@benchmark` and track performance regressions > 5%
- Prefer `StaticArrays` for small vectors

### Code Style
- 4-space indentation, lines < 92 characters
- `CamelCase` for types/modules, `snake_case` for functions/variables
- Use multiple dispatch over conditional type branches
- Document exported functions with concise docstrings