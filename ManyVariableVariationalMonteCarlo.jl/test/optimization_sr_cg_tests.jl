@testitem "optimization: SR-CG solver and SRinfo output" begin
    using ManyVariableVariationalMonteCarlo
    using Random

    # Optimization mode with SR-CG enabled via face keys
    face = FaceDefinition()
    push_definition!(face, :model, "FermionHubbard")
    push_definition!(face, :lattice, "chain")
    push_definition!(face, :L, 4)
    push_definition!(face, :nelec, 2)
    push_definition!(face, :t, 1.0)
    push_definition!(face, :U, 2.0)
    push_definition!(face, :NVMCCalMode, 0)
    push_definition!(face, :NSROptItrStep, 2)
    push_definition!(face, :NSROptItrSmp, 10)
    push_definition!(face, :NSRCG, 1)
    push_definition!(face, :NSROptCGMaxIter, 5)
    push_definition!(face, :DSROptCGTol, 1e-6)

    config = SimulationConfig(face)
    layout = ParameterLayout(2, 0, 2, 0)

    sim = VMCSimulation(config, layout; T=ComplexF64)
    initialize_simulation!(sim)
    run_parameter_optimization!(sim)

    @test length(sim.optimization_results) == 2

    outdir = mktempdir()
    output_results(sim, outdir)
    @test isfile(joinpath(outdir, "zvo_SRinfo.dat"))
    lines = readlines(joinpath(outdir, "zvo_SRinfo.dat"))
    # header + NSROptItrStep rows
    @test length(lines) == 1 + 2
end
