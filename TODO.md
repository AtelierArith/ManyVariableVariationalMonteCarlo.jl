# TODO: Missing Features from mVMC C Implementation

Based on analysis of the mVMC C implementation, the following features are supported in the original but not yet implemented in our Julia package:

## 1. Advanced Quantum Projections
- [ ] Multi-orbital quantum projections for multi-orbital Hubbard models
- [ ] Point group symmetry projections (crystal symmetry)
- [ ] Time-reversal symmetry projections
- [ ] Particle-hole symmetry projections
- [ ] Advanced symmetry operations and cubic symmetry

## 2. Unrestricted Hartree-Fock (UHF) Calculations
- [ ] ComplexUHF module for multi-orbital, multi-interaction systems
- [ ] Automatic initial wavefunction generation from UHF results
- [ ] Orbital optimization functionality
- [ ] UHF energy calculation and orbital analysis

## 3. Advanced Sampling Methods
- [ ] Burn-in sampling with controlled relaxation
- [ ] Split sampling for parallelization
- [ ] Adaptive step size adjustment
- [ ] Block updates for efficient sampling
- [ ] Autocorrelation analysis

## 4. Wannier90 Integration
- [ ] RESPACK integration for first-principles calculations
- [ ] Wannier function utilization from Wannier90
- [ ] Automatic calculation of effective interactions (U, J parameters)
- [ ] Conversion tools: respack2wan90, wout2geom

## 5. Advanced Physical Quantities
- [ ] Fourier transform tools for real-space to k-space conversion
- [ ] Detailed 1-body and 2-body Green function calculations
- [ ] Dynamic correlation functions (time-dependent)
- [ ] Spectral function calculations
- [ ] Momentum distribution analysis

## 6. Advanced Parallel Computing
- [ ] MPI parallelization for large-scale calculations
- [ ] OpenMP parallelization for multi-threading
- [ ] Hybrid parallelization (MPI + OpenMP)
- [ ] Memory optimization for large systems
- [ ] Distributed memory management

## 7. Specialized Optimization Methods
- [ ] Lanczos method integration (single Lanczos step)
- [ ] Advanced CG method implementations
- [ ] Precise stochastic reconfiguration
- [ ] Automatic parameter adjustment
- [ ] Optimization with constraints

## 8. Complete Output Format Support
- [ ] Binary output for large datasets
- [ ] HDF5 hierarchical data output
- [ ] Visualization-ready data formats
- [ ] Checkpointing for calculation restart
- [ ] Advanced file I/O with multiple formats

## 9. Specialized Lattices and Models
- [ ] Pyrochlore lattice (3D frustrated)
- [ ] Kagome lattice (2D frustrated)
- [ ] Kitaev model (quantum spin liquid)
- [ ] Multi-orbital models
- [ ] Complex interaction models

## 10. Tools and Utilities
- [ ] greenr2k: Green function k-space conversion
- [ ] respack2wan90: RESPACK to Wannier90 conversion
- [ ] wout2geom: Geometry information extraction
- [ ] gen_frmsf: Frustration information generation
- [ ] Fourier analysis tools

## 11. Advanced Wavefunctions
- [ ] Backflow corrections for electron correlation
- [ ] Advanced RBM implementations
- [ ] Composite wavefunctions (multiple correlation factors)
- [ ] Frozen spin functionality
- [ ] Enhanced Jastrow factors

## 12. Numerical Computing Optimization
- [ ] BLIS library integration for high-performance linear algebra
- [ ] ScaLAPACK for parallel linear algebra
- [ ] Memory management for large systems
- [ ] Numerical stability improvements
- [ ] High-precision calculations

## Priority Implementation Order
1. **High Priority**: UHF calculations, Wannier90 integration, advanced sampling
2. **Medium Priority**: Advanced quantum projections, specialized lattices
3. **Low Priority**: Tools and utilities, advanced output formats

## Notes
- Many features require external dependencies (MPI, BLIS, ScaLAPACK)
- Some features may need significant architectural changes
- Consider implementing core features first, then advanced features
- Maintain compatibility with existing mVMC workflow
