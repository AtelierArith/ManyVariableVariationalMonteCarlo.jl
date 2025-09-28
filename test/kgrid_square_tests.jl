@testitem "k-grid: square Lx×Ly yields Lx*Ly rows and 2D format" begin
    using ManyVariableVariationalMonteCarlo

    Lx, Ly = 4, 3
    face = FaceDefinition()
    push_definition!(face, :model, "FermionHubbard")
    push_definition!(face, :lattice, "square")
    push_definition!(face, :L, Lx)  # In this codebase, L may be used; geometry reads L and W
    push_definition!(face, :W, Ly)
    push_definition!(face, :nelec, 6)
    push_definition!(face, :NVMCCalMode, 1)
    push_definition!(face, :NVMCSample, 10)

    config = SimulationConfig(face)
    layout = ParameterLayout(2, 0, 2, 0)

    sim = VMCSimulation(config, layout)
    initialize_simulation!(sim)
    run_physics_calculation!(sim)

    outdir = mktempdir()
    output_results(sim, outdir)

    # Read struct lines
    data_struct = filter(
        l -> !isempty(l) && !startswith(l, "#"),
        readlines(joinpath(outdir, "zvo_struct.dat")),
    )
    @test length(data_struct) == Lx * Ly
    # Each data line: kx ky ReSs ImSs ReSn ImSn (6 tokens)
    parts = split(strip(first(data_struct)))
    @test length(parts) == 6

    # Read momentum lines
    data_nk = filter(
        l -> !isempty(l) && !startswith(l, "#"),
        readlines(joinpath(outdir, "zvo_momentum.dat")),
    )
    @test length(data_nk) == Lx * Ly
    parts_nk = split(strip(first(data_nk)))
    @test length(parts_nk) == 4  # kx ky Re nk Im nk
end
