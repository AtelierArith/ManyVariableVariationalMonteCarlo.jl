# ManyVariableVariationalMonteCarlo.jl

[![Build Status](https://github.com/AtelierArith/ManyVariableVariationalMonteCarlo.jl/workflows/CI/badge.svg)](https://github.com/AtelierArith/ManyVariableVariationalMonteCarlo.jl/actions)

A Julia implementation of many-variable variational Monte Carlo (VMC) method for quantum many-body systems, inspired by the [mVMC](https://github.com/issp-center-dev/mVMC) C package.

## ✨ Features

- **Standard Lattice Models**: Built-in support for chain, square, triangular, honeycomb, kagome, and ladder lattices
- **Quantum Models**: Hubbard, Heisenberg, Kondo models with customizable parameters
- **VMC Workflow**: Complete simulation pipeline with parameter optimization and physics calculations
- **Wave Functions**: Slater determinants, Jastrow correlations, and RBM (Restricted Boltzmann Machine) support
- **High Performance**: Optimized linear algebra with thread-local workspaces and efficient updates
- **mVMC Compatibility**: Compatible I/O and workflow with the original mVMC package

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/AtelierArith/ManyVariableVariationalMonteCarlo.jl.git
cd ManyVariableVariationalMonteCarlo.jl
git submodule init
git submodule update
cd ManyVariableVariationalMonteCarlo.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

### Basic Usage

```julia
using ManyVariableVariationalMonteCarlo

# Create a 1D Hubbard chain
config = SimulationConfig(
    model="FermionHubbard",
    lattice="chain", 
    L=8,
    nelec=4,
    t=1.0,
    U=2.0
)

# Set up variational parameters
layout = ParameterLayout(config.nsites, 0, config.nsites, 0)

# Run VMC simulation
sim = VMCSimulation(config, layout; T=ComplexF64)
run_simulation!(sim)

# View results
print_simulation_summary(sim)
```

### Using StdFace Interface

```julia
using ManyVariableVariationalMonteCarlo

# Create standard lattice models easily
ham, geom, config = stdface_square(4, 4, "Hubbard"; t=1.0, U=4.0)

# Display lattice information
lattice_summary(geom)
hamiltonian_summary(ham)
```

## 📖 Documentation

### Running Examples

The package includes comprehensive examples demonstrating various quantum models:

```bash
# Basic examples
julia --project examples/01_1D_Hubbard.jl          # 1D Hubbard model
julia --project examples/02_2D_Heisenberg.jl       # 2D Heisenberg model
julia --project examples/03_basic_workflow.jl      # Complete VMC workflow

# Advanced models  
julia --project examples/06_1D_Kondo.jl            # Kondo model
julia --project examples/07_2D_ExHubbard.jl        # Extended Hubbard model
julia --project examples/08_2D_J1J2_Heisenberg.jl  # Frustrated J1-J2 model
julia --project examples/09_2D_AttractiveHubbard.jl # Attractive Hubbard model

# StdFace integration
julia --project examples/04_stdface_lattices.jl     # Standard lattices
julia --project examples/05_stdface_vmc_integration.jl # VMC integration
```

### Key Components

#### 1. **Simulation Configuration**
```julia
# Manual configuration
config = SimulationConfig(
    model="FermionHubbard",
    lattice="square",
    Lx=4, Ly=4,
    nelec=8,
    t=1.0, U=4.0
)

# Using FaceDefinition (StdFace-style)
face = FaceDefinition()
push_definition!(face, :model, "FermionHubbard")
push_definition!(face, :lattice, "square") 
push_definition!(face, :Lx, 4)
push_definition!(face, :Ly, 4)
config = SimulationConfig(face)
```

#### 2. **Hamiltonian Construction**
```julia
# Automatic from configuration
ham = create_hubbard_hamiltonian(config.nsites, config.nelec, config.t, config.u)

# Using StdFace
ham, geom, config = stdface_square(4, 4, "Hubbard"; t=1.0, U=4.0)

# Manual construction
ham = Hamiltonian{ComplexF64}(nsites=16, nelec=8)
add_transfer!(ham, 1, 2, -1.0)  # Hopping
add_coulomb_intra!(ham, 1, 4.0) # On-site interaction
```

#### 3. **Wave Function Components**
```julia
# Slater determinant
slater = SlaterDeterminant{ComplexF64}(nsites, nelec)
initialize_slater!(slater, electron_config)

# Jastrow correlations
jastrow = JastrowFactor{Float64}()
add_gutzwiller_parameter!(jastrow, 1, 0.5)

# RBM neural quantum states  
rbm = RBMNetwork{ComplexF64}(nsites, nhidden)
initialize_rbm!(rbm, weights, biases)
```

#### 4. **VMC Simulation Modes**
```julia
# Parameter optimization mode
config.NVMCCalMode = 0
config.NSROptItrStep = 100     # Optimization steps
config.NSROptItrSmp = 1000     # Samples per step

# Physics calculation mode  
config.NVMCCalMode = 1
config.NVMCSample = 10000      # Total samples
config.NVMCInterval = 1        # Sampling interval
```

### Available Lattice Types

| Lattice | Dimensions | StdFace Function |
|---------|------------|------------------|
| Chain | 1D | `stdface_chain(L, model; ...)` |
| Square | 2D | `stdface_square(Lx, Ly, model; ...)` |
| Triangular | 2D | `stdface_triangular(Lx, Ly, model; ...)` |
| Honeycomb | 2D | `stdface_honeycomb(Lx, Ly, model; ...)` |
| Kagome | 2D | `stdface_kagome(Lx, Ly, model; ...)` |
| Ladder | 2D | `stdface_ladder(Lx, Ly, model; ...)` |

### Supported Models

| Model | Description | Parameters |
|-------|-------------|------------|
| `"FermionHubbard"` | Hubbard model | `t`, `U`, `V`, `t'` |
| `"Spin"` | Heisenberg model | `J`, `J'`, `Jz` |
| `"Kondo"` | Kondo lattice | `t`, `J`, `U` |

## 🔧 Advanced Usage

### Loading mVMC Configuration Files

```julia
# Load from mVMC namelist.def
config_dict = load_vmc_configuration("path/to/namelist.def")
sim = VMCSimulation(config_dict)
run_simulation!(sim)
```

### Custom Optimization

```julia
# Stochastic reconfiguration method
sr = StochasticReconfiguration{ComplexF64}(nparam)
config.NSROptCGMaxIter = 100   # CG solver iterations
config.DSROptCGTol = 1e-6      # CG tolerance

# Custom optimization loop
for step in 1:config.NSROptItrStep
    # Sample and compute gradients
    run_vmc_sampling!(sim)
    compute_overlap_matrix!(sr, sim)
    compute_force_vector!(sr, sim)
    
    # Update parameters
    solve_sr_equations!(sr)
    update_parameters!(sim, sr.delta_params)
end
```

### Output Files

The package generates mVMC-compatible output files:

| File | Content |
|------|---------|
| `zvo_result.dat` | Main results (energy, variance, etc.) |
| `zvo_energy.dat` | Energy time series |
| `zvo_accept.dat` | Acceptance rate time series |
| `zvo_corr.dat` | Correlation functions |
| `zvo_struct.dat` | Structure factors |
| `zvo_momentum.dat` | Momentum distribution |
| `zvo_cisajs.dat` | Green's functions |

## 🧪 Testing

```bash
# Run all tests
julia --project -e 'using Pkg; Pkg.test()'

# Run specific test files
julia --project test/hamiltonian_tests.jl
julia --project test/stdface_tests.jl
julia --project test/vmcmain_tests.jl
```

## 📊 Performance

The Julia implementation provides:
- **Thread-local workspaces** for parallel Monte Carlo sampling
- **Optimized BLAS operations** with efficient memory management
- **Sherman-Morrison updates** for fast matrix inverse updates
- **Pfaffian algorithms** with numerical stability controls

Typical performance for a 4×4 Hubbard model (1000 samples):
- Parameter optimization: ~1-2 seconds per iteration
- Physics calculation: ~0.5-1 second per 1000 samples

## 🤝 Contributing

We welcome contributions! Please see our development workflow:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

### Development Setup

```bash
git clone https://github.com/AtelierArith/ManyVariableVariationalMonteCarlo.jl.git
cd ManyVariableVariationalMonteCarlo.jl
git submodule update --init --recursive
cd ManyVariableVariationalMonteCarlo.jl
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

## 📚 References

- **mVMC**: [Many-variable Variational Monte Carlo](https://github.com/issp-center-dev/mVMC)
- **mVMC Tutorial**: Hands-on materials for quantum many-body calculations
- **VMC Method**: Variational Monte Carlo for quantum many-body systems

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This work is inspired by and builds upon the excellent [mVMC](https://github.com/issp-center-dev/mVMC) package developed by the ISSP (Institute for Solid State Physics, University of Tokyo) team.
