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

## mVMC C Reference Analysis

### Source Code Structure (68 C files total)
The mVMC C implementation is organized into focused modules:

#### Core Directories:
- `src/mVMC/`: Main VMC implementation (57 files)
- `src/common/`: Shared utilities (memory management, BLAS wrappers)
- `src/ComplexUHF/`: Unrestricted Hartree-Fock solver for initial states
- `src/StdFace/`: Standard lattice model generator
- `src/pfupdates/`: Pfaffian update algorithms
- `src/sfmt/`: SIMD-oriented Fast Mersenne Twister RNG

#### Key Implementation Components:
1. **Main Program Flow** (`vmcmain.c`):
   - Parameter optimization mode vs. expectation value calculation
   - MPI parallelization setup
   - File I/O coordination

2. **Core VMC Algorithm** (`vmccal.c`, `vmccal_fsz.c`):
   - Monte Carlo sampling loop
   - Metropolis acceptance/rejection
   - Observable measurement

3. **Wavefunction Components**:
   - `slater.c`/`slater_fsz.c`: Slater determinant
   - `rbm.c`: Restricted Boltzmann Machine neural network
   - `projection.c`: Quantum number projection

4. **Monte Carlo Updates**:
   - `pfupdate*.c`: Pfaffian matrix updates (single/two electron)
   - `locgrn*.c`: Local Green function calculations
   - `lslocgrn*.c`: Large-scale local Green functions

5. **Optimization** (`stcopt*.c`):
   - Stochastic reconfiguration (SR) method
   - Conjugate gradient solver
   - Parameter gradient calculation

6. **Hamiltonian** (`calham*.c`):
   - Energy calculation
   - Matrix element evaluation
   - Different precision variants (real/complex, single/double)

7. **Configuration Management** (`readdef.c`, `parameter.c`):
   - Input file parsing
   - Lattice model setup
   - Memory allocation

## Julia Port Implementation Roadmap

### Phase 1: Core Infrastructure
1. **Configuration System** (src/config.jl enhancement):
   - Port `readdef.c` parameter parsing (94KB, largest file)
   - Implement `parameter.c` validation logic
   - Add support for all mVMC input formats

2. **Memory Management** (src/memory.jl):
   - Port `setmemory.c` allocation strategies
   - Implement workspace management from `workspace.c`
   - Add memory pooling for performance

3. **Random Number Generation** (src/rng.jl):
   - Port SFMT implementation or use Julia's MersenneTwister
   - Ensure reproducible seeds for testing
   - Implement parallel RNG streams

### Phase 2: Mathematical Foundation
4. **Linear Algebra Backend** (src/linalg.jl):
   - Port BLAS/LAPACK wrappers from `matrix.c`
   - Implement Pfaffian calculations (`pfupdate*.c` family)
   - Add optimized complex number operations

5. **Green Functions** (src/greens.jl):
   - Port local Green function calculations (`locgrn*.c`)
   - Implement large-scale variants (`lslocgrn*.c`)
   - Add caching and update strategies

6. **Quantum Projections** (src/projections.jl):
   - Port `projection.c` quantum number projection
   - Implement Gauss-Legendre quadrature (`gauleg.c`)
   - Add symmetry operations

### Phase 3: Wavefunction Components
7. **Slater Determinants** (src/slater.jl):
   - Port `slater.c` determinant evaluation
   - Implement fast updates and ratios
   - Add support for frozen-spin variants (`slater_fsz.c`)

8. **Neural Networks** (src/rbm.jl):
   - Port `rbm.c` Restricted Boltzmann Machine
   - Implement efficient gradient calculations
   - Add variational parameter management

9. **Jastrow Factors** (src/jastrow.jl):
   - Implement correlation factors
   - Add optimized update algorithms
   - Support various correlation forms

### Phase 4: Monte Carlo Engine
10. **Sampling Core** (src/sampler.jl):
    - Port main sampling loop from `vmccal.c`
    - Implement Metropolis-Hastings algorithm
    - Add thermalization and autocorrelation tracking

11. **Update Algorithms** (src/updates.jl):
    - Port single-electron updates
    - Implement two-electron updates (`pfupdate_two*.c`)
    - Add exchange hopping updates

12. **Observable Measurement** (src/observables.jl):
    - Port energy calculation (`calham*.c`)
    - Implement correlation function measurement
    - Add custom observable framework

### Phase 5: Optimization
13. **Stochastic Reconfiguration** (src/optimization.jl):
    - Port SR method from `stcopt*.c`
    - Implement conjugate gradient solver
    - Add diagonal preconditioning

14. **Parameter Management** (src/parameters.jl enhancement):
    - Extend existing `ParameterSet` types
    - Add automatic differentiation support
    - Implement parameter history tracking

### Phase 6: I/O and Analysis
15. **File I/O** (src/io.jl enhancement):
    - Port binary file formats
    - Implement checkpoint/restart capability
    - Add HDF5 output support

16. **Analysis Tools** (src/analysis.jl):
    - Port averaging routines (`average.c`, `avevar.c`)
    - Implement statistical analysis
    - Add visualization helpers

### Phase 7: Advanced Features
17. **Parallelization** (src/parallel.jl):
    - Implement distributed computing support
    - Add GPU acceleration hooks
    - Port MPI coordination patterns

18. **Standard Models** (src/models.jl):
    - Port StdFace lattice generator
    - Implement common model definitions
    - Add model validation

### Phase 8: Integration & Validation
19. **ComplexUHF Integration** (src/uhf.jl):
    - Port unrestricted Hartree-Fock solver
    - Implement initial state generation
    - Add trial wavefunction optimization

20. **Comprehensive Testing**:
    - Port all test cases from C implementation
    - Add performance benchmarks vs. C code
    - Implement regression testing

### Implementation Priority:
1. **High Priority**: Core infrastructure, mathematical foundation, wavefunction components
2. **Medium Priority**: Monte Carlo engine, optimization, I/O
3. **Low Priority**: Advanced features, ComplexUHF, comprehensive testing

### Estimated Implementation Effort:
- **Phase 1-3**: ~2-3 months (foundational components)
- **Phase 4-6**: ~2-3 months (core VMC functionality)
- **Phase 7-8**: ~1-2 months (advanced features & validation)

Total: **5-8 months** for complete port with testing and optimization.