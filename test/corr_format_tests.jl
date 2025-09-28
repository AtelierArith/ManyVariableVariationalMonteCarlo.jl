@testitem "corr format: zvo_corr has pair columns" begin
    using ManyVariableVariationalMonteCarlo

    face = FaceDefinition()
    push_definition!(face, :model, "FermionHubbard")
    push_definition!(face, :lattice, "chain")
    push_definition!(face, :L, 6)
    push_definition!(face, :nelec, 4)
    push_definition!(face, :NVMCCalMode, 1)
    push_definition!(face, :NVMCSample, 10)

    config = SimulationConfig(face)
    layout = ParameterLayout(2, 0, 2, 0)

    sim = VMCSimulation(config, layout)
    initialize_simulation!(sim)
    run_physics_calculation!(sim)

    outdir = mktempdir()
    output_results(sim, outdir)

    # Find first data line in zvo_corr.dat and check columns
    for line in eachline(joinpath(outdir, "zvo_corr.dat"))
        if isempty(line) || startswith(line, "#")
            ;
            continue;
        end
        parts = split(strip(line))
        # distance + 6 numeric columns: total 7 tokens
        @test length(parts) == 7
        break
    end
end
