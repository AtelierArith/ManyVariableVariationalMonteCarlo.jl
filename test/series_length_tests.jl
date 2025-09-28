@testitem "series lengths: energy and acceptance match samples" begin
    using ManyVariableVariationalMonteCarlo

    face = FaceDefinition()
    push_definition!(face, :model, "FermionHubbard")
    push_definition!(face, :lattice, "chain")
    push_definition!(face, :L, 6)
    push_definition!(face, :nelec, 4)
    push_definition!(face, :NVMCCalMode, 1)
    push_definition!(face, :NVMCSample, 30)

    config = SimulationConfig(face)
    layout = ParameterLayout(2, 0, 2, 0)

    sim = VMCSimulation(config, layout)
    initialize_simulation!(sim)
    run_physics_calculation!(sim)

    outdir = mktempdir()
    output_results(sim, outdir)

    # zvo_energy.dat
    e_lines = filter(l -> !isempty(l) && !startswith(l, "#"), readlines(joinpath(outdir, "zvo_energy.dat")))
    @test length(e_lines) == 30

    # zvo_accept.dat
    a_lines = filter(l -> !isempty(l) && !startswith(l, "#"), readlines(joinpath(outdir, "zvo_accept.dat")))
    @test length(a_lines) == 30
end
