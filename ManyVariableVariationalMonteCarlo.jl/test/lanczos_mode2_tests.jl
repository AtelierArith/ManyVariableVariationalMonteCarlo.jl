@testitem "lanczos: NLanczosMode=2 writes zvo_ls_cisajs.dat" begin
    using ManyVariableVariationalMonteCarlo

    face = FaceDefinition()
    push_definition!(face, :model, "FermionHubbard")
    push_definition!(face, :lattice, "chain")
    push_definition!(face, :L, 4)
    push_definition!(face, :nelec, 2)
    push_definition!(face, :NVMCCalMode, 1)
    push_definition!(face, :NVMCSample, 10)
    push_definition!(face, :NLanczosMode, 2)
    # enable Local 1-body
    push_definition!(face, :OneBodyG, true)

    config = SimulationConfig(face)
    layout = ParameterLayout(2, 0, 2, 0)

    sim = VMCSimulation(config, layout)
    initialize_simulation!(sim)
    run_physics_calculation!(sim)

    outdir = mktempdir()
    output_results(sim, outdir)

    @test isfile(joinpath(outdir, "zvo_ls_cisajs.dat"))
end
