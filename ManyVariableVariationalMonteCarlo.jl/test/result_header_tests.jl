@testitem "zvo_result header includes physics summary" begin
    using ManyVariableVariationalMonteCarlo

    face = FaceDefinition()
    push_definition!(face, :model, "FermionHubbard")
    push_definition!(face, :lattice, "chain")
    push_definition!(face, :L, 4)
    push_definition!(face, :nelec, 2)
    push_definition!(face, :NVMCCalMode, 1)
    push_definition!(face, :NVMCSample, 5)

    config = SimulationConfig(face)
    layout = ParameterLayout(2, 0, 2, 0)
    sim = VMCSimulation(config, layout)
    initialize_simulation!(sim)
    run_physics_calculation!(sim)

    outdir = mktempdir()
    output_results(sim, outdir)

    lines = readlines(joinpath(outdir, "zvo_result.dat"))
    @test any(contains.(lines, "# VMC Physics Calculation Results"))
    @test any(contains.(lines, "# Energy:"))
    @test any(contains.(lines, "# Number of Samples:"))
end
