@testitem "vmc observables: correlations and output" begin
    using ManyVariableVariationalMonteCarlo
    using Random
    using Printf
    import ManyVariableVariationalMonteCarlo: compute_equal_time_correlations

    # Small chain system
    face = FaceDefinition()
    push_definition!(face, :model, "FermionHubbard")
    push_definition!(face, :lattice, "chain")
    push_definition!(face, :L, 6)
    push_definition!(face, :nelec, 4)
    push_definition!(face, :t, 1.0)
    push_definition!(face, :U, 2.0)
    push_definition!(face, :NVMCCalMode, 1)
    push_definition!(face, :NVMCSample, 20)

    config = SimulationConfig(face)
    layout = ParameterLayout(2, 0, 2, 0)

    sim = VMCSimulation(config, layout; T=ComplexF64)
    initialize_simulation!(sim)
    run_physics_calculation!(sim)

    @test haskey(sim.physics_results, "spin_correlation")
    @test haskey(sim.physics_results, "density_correlation")
    spin = sim.physics_results["spin_correlation"]
    dens = sim.physics_results["density_correlation"]
    @test length(spin) >= 1 && length(dens) >= 1

    # Check computation helper yields consistent sizes
    corrs = compute_equal_time_correlations(sim.vmc_state; max_distance=3)
    @test length(corrs.spin) == 4
    @test length(corrs.density) == 4

    # Check output files are written
    outdir = mktempdir()
    output_results(sim, outdir)
    @test isfile(joinpath(outdir, "zvo_result.dat"))
    # Correlation file should exist as we filled arrays
    @test isfile(joinpath(outdir, "zvo_corr.dat"))
    # Energy time series file should exist
    @test isfile(joinpath(outdir, "zvo_energy.dat"))
    # Acceptance time series
    @test isfile(joinpath(outdir, "zvo_accept.dat"))
    # Structure factors and momentum distribution files
    @test isfile(joinpath(outdir, "zvo_struct.dat"))
    @test isfile(joinpath(outdir, "zvo_momentum.dat"))
    # One-body Green function snapshot file
    @test isfile(joinpath(outdir, "zvo_cisajs.dat"))
    # Pair correlation included in zvo_corr.dat columns
end
