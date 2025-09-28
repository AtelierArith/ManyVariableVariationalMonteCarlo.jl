@testsetup module VMCTestSetup
using ManyVariableVariationalMonteCarlo
using Random
using LinearAlgebra
end

@testitem "VMCSimulation basic construction" begin
    # Create a simple configuration
    face = FaceDefinition()
    push_definition!(face, :model, "FermionHubbard")
    push_definition!(face, :lattice, "chain")
    push_definition!(face, :L, 4)
    push_definition!(face, :nelec, 4)
    push_definition!(face, :t, 1.0)
    push_definition!(face, :U, 2.0)
    push_definition!(face, :NVMCCalMode, 0)
    push_definition!(face, :NSROptItrStep, 10)
    push_definition!(face, :NVMCSample, 100)

    config = SimulationConfig(face)
    layout = ParameterLayout(4, 0, 8, 0)  # Some projection and Slater parameters

    sim = VMCSimulation(config, layout)

    @test sim.config.nsites == 4
    @test sim.config.nelec == 4
    @test sim.mode == PARAMETER_OPTIMIZATION
    @test length(sim.parameters) == 12
    @test sim.vmc_state === nothing  # Not initialized yet
end

@testitem "VMCSimulation initialization" begin
    # Create configuration
    face = FaceDefinition()
    push_definition!(face, :model, "FermionHubbard")
    push_definition!(face, :lattice, "chain")
    push_definition!(face, :L, 4)
    push_definition!(face, :nelec, 4)
    push_definition!(face, :t, 1.0)
    push_definition!(face, :U, 2.0)
    push_definition!(face, :NVMCCalMode, 1)  # Physics calculation mode

    config = SimulationConfig(face)
    layout = ParameterLayout(2, 0, 4, 0)

    sim = VMCSimulation(config, layout)
    initialize_simulation!(sim)

    @test sim.vmc_state !== nothing
    @test sim.vmc_state.n_electrons == 4
    @test sim.vmc_state.n_sites == 4
    @test sim.slater_det !== nothing
    @test sim.jastrow_factor !== nothing
    @test sim.workspace !== nothing
    @test sim.mode == PHYSICS_CALCULATION
end

@testitem "VMCSimulation parameter optimization mode" begin
    # Create simple system for optimization
    face = FaceDefinition()
    push_definition!(face, :model, "FermionHubbard")
    push_definition!(face, :lattice, "chain")
    push_definition!(face, :L, 4)
    push_definition!(face, :nelec, 2)
    push_definition!(face, :t, 1.0)
    push_definition!(face, :U, 2.0)
    push_definition!(face, :NVMCCalMode, 0)
    push_definition!(face, :NSROptItrStep, 3)  # Short optimization
    push_definition!(face, :NSROptItrSmp, 10)
    push_definition!(face, :NVMCSample, 50)

    config = SimulationConfig(face)
    layout = ParameterLayout(2, 0, 2, 0)

    sim = VMCSimulation(config, layout; T = ComplexF64)
    initialize_simulation!(sim)

    # Run short optimization
    run_parameter_optimization!(sim)

    @test length(sim.optimization_results) == 3
    @test haskey(sim.optimization_results[1], "energy")
    @test haskey(sim.optimization_results[1], "iteration")
    @test sim.optimization_results[1]["iteration"] == 1
    @test sim.optimization_results[end]["iteration"] == 3
end

@testitem "VMCSimulation physics calculation mode" begin
    # Create system for physics calculation
    face = FaceDefinition()
    push_definition!(face, :model, "FermionHubbard")
    push_definition!(face, :lattice, "chain")
    push_definition!(face, :L, 4)
    push_definition!(face, :nelec, 2)
    push_definition!(face, :t, 1.0)
    push_definition!(face, :U, 2.0)
    push_definition!(face, :NVMCCalMode, 1)
    push_definition!(face, :NVMCSample, 50)

    config = SimulationConfig(face)
    layout = ParameterLayout(2, 0, 2, 0)

    sim = VMCSimulation(config, layout; T = ComplexF64)
    initialize_simulation!(sim)

    # Run physics calculation
    run_physics_calculation!(sim)

    @test !isempty(sim.physics_results)
    @test haskey(sim.physics_results, "energy_mean")
    @test haskey(sim.physics_results, "energy_std")
    @test haskey(sim.physics_results, "double_occupation")
    @test haskey(sim.physics_results, "acceptance_rate")
    @test sim.physics_results["n_samples"] == 50
end

@testitem "VMCSimulation full workflow" begin
    # Test complete simulation workflow
    face = FaceDefinition()
    push_definition!(face, :model, "FermionHubbard")
    push_definition!(face, :lattice, "chain")
    push_definition!(face, :L, 6)
    push_definition!(face, :nelec, 3)
    push_definition!(face, :t, 1.0)
    push_definition!(face, :U, 4.0)
    push_definition!(face, :NVMCCalMode, 0)
    push_definition!(face, :NSROptItrStep, 2)
    push_definition!(face, :NVMCSample, 30)

    config = SimulationConfig(face)
    layout = ParameterLayout(3, 0, 6, 0)

    sim = VMCSimulation(config, layout)

    # Test full workflow
    run_simulation!(sim)

    @test sim.vmc_state !== nothing
    @test length(sim.optimization_results) == 2

    # Test summary output
    print_simulation_summary(sim)  # Should not error
end

@testitem "VMCSimulation with RBM network" begin
    # Test simulation with RBM parameters
    face = FaceDefinition()
    push_definition!(face, :model, "FermionHubbard")
    push_definition!(face, :lattice, "chain")
    push_definition!(face, :L, 4)
    push_definition!(face, :nelec, 2)
    push_definition!(face, :NVMCCalMode, 1)
    push_definition!(face, :NVMCSample, 20)

    config = SimulationConfig(face)
    layout = ParameterLayout(2, 10, 4, 0)  # Include RBM parameters

    sim = VMCSimulation(config, layout)
    initialize_simulation!(sim)

    @test sim.rbm_network !== nothing
    @test sim.slater_det !== nothing
    @test sim.jastrow_factor !== nothing

    # Run short physics calculation
    run_physics_calculation!(sim)

    @test !isempty(sim.physics_results)
end

@testitem "VMCSimulation double occupation measurement" begin
    # Test double occupation calculation
    face = FaceDefinition()
    push_definition!(face, :L, 4)
    push_definition!(face, :nelec, 4)

    config = SimulationConfig(face)
    layout = ParameterLayout(2, 0, 2, 0)

    sim = VMCSimulation(config, layout)
    initialize_simulation!(sim)

    # Manually set up a configuration with known double occupation
    # Place 2 electrons on first 2 sites (one up, one down each)
    sim.vmc_state.electron_positions[1:4] = [1, 2, 1, 2]

    double_occ = measure_double_occupation(sim.vmc_state)
    @test double_occ ≈ 0.5  # 2 doubly occupied sites out of 4
end

@testitem "VMCSimulation error handling" begin
    # Test error handling for invalid configurations
    face = FaceDefinition()
    push_definition!(face, :NVMCCalMode, 99)  # Invalid mode value

    config = SimulationConfig(face)
    layout = ParameterLayout(1, 0, 1, 0)

    # This should throw an error due to invalid enum value
    @test_throws ArgumentError VMCSimulation(config, layout)
end
