@testitem "k-grid: chain k values are 2π n/L" begin
    using ManyVariableVariationalMonteCarlo

    L = 8
    face = FaceDefinition()
    push_definition!(face, :model, "FermionHubbard")
    push_definition!(face, :lattice, "chain")
    push_definition!(face, :L, L)
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

    # Parse k values from struct file
    ks = Float64[]
    for line in eachline(joinpath(outdir, "zvo_struct.dat"))
        if isempty(line) || startswith(line, "#")
            ;
            continue;
        end
        parts = split(strip(line))
        # First column is k
        push!(ks, parse(Float64, parts[1]))
    end
    @test length(ks) == L

    # Expected k: 2π n / L
    expected = [2pi * n / L for n = 0:(L-1)]
    for i = 1:L
        @test isapprox(ks[i], expected[i]; atol = 1e-9)
    end
end
