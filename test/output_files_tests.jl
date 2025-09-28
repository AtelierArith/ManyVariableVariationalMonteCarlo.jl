@testitem "vmc outputs: zvo_out and 4-body placeholders" begin
    using ManyVariableVariationalMonteCarlo
    using Random

    # Prepare a small system (physics mode)
    face = FaceDefinition()
    push_definition!(face, :model, "FermionHubbard")
    push_definition!(face, :lattice, "chain")
    push_definition!(face, :L, 6)
    push_definition!(face, :nelec, 4)
    push_definition!(face, :t, 1.0)
    push_definition!(face, :U, 2.0)
    push_definition!(face, :NVMCCalMode, 1)
    push_definition!(face, :NVMCSample, 20)
    # Enable flush for coverage
    push_definition!(face, :FlushFile, true)

    config = SimulationConfig(face)
    layout = ParameterLayout(2, 0, 2, 0)

    sim = VMCSimulation(config, layout; T = ComplexF64)
    initialize_simulation!(sim)
    run_physics_calculation!(sim)

    outdir = mktempdir()
    output_results(sim, outdir)

    # Check the presence of all expected output files
    @test isfile(joinpath(outdir, "zvo_result.dat"))
    @test isfile(joinpath(outdir, "zvo_out.dat"))
    @test isfile(joinpath(outdir, "zvo_corr.dat"))
    @test isfile(joinpath(outdir, "zvo_energy.dat"))
    @test isfile(joinpath(outdir, "zvo_accept.dat"))
    @test isfile(joinpath(outdir, "zvo_struct.dat"))
    @test isfile(joinpath(outdir, "zvo_momentum.dat"))
    @test isfile(joinpath(outdir, "zvo_cisajs.dat"))
    # Placeholders
    @test isfile(joinpath(outdir, "zvo_cisajscktaltex.dat"))
    @test isfile(joinpath(outdir, "zvo_cisajscktalt.dat"))

    # Sanity check zvo_out.dat has one data line
    lines = readlines(joinpath(outdir, "zvo_out.dat"))
    @test length(lines) >= 2  # header + at least one row
end
