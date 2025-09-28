@testitem "lanczos: zvo_ls_* outputs when NLanczosMode>0" begin
    using ManyVariableVariationalMonteCarlo

    face = FaceDefinition()
    push_definition!(face, :model, "FermionHubbard")
    push_definition!(face, :lattice, "chain")
    push_definition!(face, :L, 6)
    push_definition!(face, :nelec, 4)
    push_definition!(face, :NVMCCalMode, 1)
    push_definition!(face, :NVMCSample, 20)
    push_definition!(face, :NLanczosMode, 1)

    config = SimulationConfig(face)
    layout = ParameterLayout(2, 0, 2, 0)

    sim = VMCSimulation(config, layout)
    initialize_simulation!(sim)
    run_physics_calculation!(sim)

    outdir = mktempdir()
    output_results(sim, outdir)

    @test isfile(joinpath(outdir, "zvo_ls_result.dat"))
    @test isfile(joinpath(outdir, "zvo_ls_alpha_beta.dat"))

    lines = readlines(joinpath(outdir, "zvo_ls_result.dat"))
    @test length(lines) >= 2
end
