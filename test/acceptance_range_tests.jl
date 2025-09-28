@testitem "acceptance series in [0,1]" begin
    using ManyVariableVariationalMonteCarlo

    face = FaceDefinition()
    push_definition!(face, :model, "FermionHubbard")
    push_definition!(face, :lattice, "chain")
    push_definition!(face, :L, 6)
    push_definition!(face, :nelec, 4)
    push_definition!(face, :NVMCCalMode, 1)
    push_definition!(face, :NVMCSample, 20)

    config = SimulationConfig(face)
    layout = ParameterLayout(2, 0, 2, 0)
    sim = VMCSimulation(config, layout)
    initialize_simulation!(sim)
    run_physics_calculation!(sim)

    outdir = mktempdir()
    output_results(sim, outdir)

    for line in eachline(joinpath(outdir, "zvo_accept.dat"))
        if isempty(line) || startswith(line, "#")
            ;
            continue;
        end
        parts = split(strip(line))
        a = parse(Float64, parts[end])
        @test 0.0 <= a <= 1.0
    end
end
