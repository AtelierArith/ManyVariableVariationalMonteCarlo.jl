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

## TODOs

Refer ./TODO.md to learn more.
