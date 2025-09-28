@testitem "k-grid numeric: square k values and finite data" begin
    using ManyVariableVariationalMonteCarlo

    Lx, Ly = 3, 5
    face = FaceDefinition()
    push_definition!(face, :model, "FermionHubbard")
    push_definition!(face, :lattice, "square")
    push_definition!(face, :L, Lx)
    push_definition!(face, :W, Ly)
    push_definition!(face, :nelec, 6)
    push_definition!(face, :NVMCCalMode, 1)
    push_definition!(face, :NVMCSample, 12)

    config = SimulationConfig(face)
    layout = ParameterLayout(2, 0, 2, 0)

    sim = VMCSimulation(config, layout)
    initialize_simulation!(sim)
    run_physics_calculation!(sim)

    outdir = mktempdir()
    output_results(sim, outdir)

    # Build expected k-grid ordering: nx outer, ny inner (as in implementation)
    expected_k = [(2pi * nx / Lx, 2pi * ny / Ly) for nx in 0:(Lx-1), ny in 0:(Ly-1)]
    expected_k = collect(Iterators.flatten(expected_k))

    # zvo_struct.dat: kx ky ReSs ImSs ReSn ImSn
    lines_struct = filter(l -> !isempty(l) && !startswith(l, "#"), readlines(joinpath(outdir, "zvo_struct.dat")))
    @test length(lines_struct) == Lx * Ly
    for (idx, line) in enumerate(lines_struct)
        parts = split(strip(line))
        @test length(parts) == 6
        kx = parse(Float64, parts[1])
        ky = parse(Float64, parts[2])
        @test isapprox(kx, expected_k[idx][1]; atol=1e-9)
        @test isapprox(ky, expected_k[idx][2]; atol=1e-9)
        # values are finite and not absurdly large
        vals = parse.(Float64, parts[3:end])
        @test all(isfinite, vals)
        @test all(abs.(vals) .< 1e9)
    end

    # zvo_momentum.dat: kx ky Re[nk] Im[nk]
    lines_nk = filter(l -> !isempty(l) && !startswith(l, "#"), readlines(joinpath(outdir, "zvo_momentum.dat")))
    @test length(lines_nk) == Lx * Ly
    for (idx, line) in enumerate(lines_nk)
        parts = split(strip(line))
        @test length(parts) == 4
        kx = parse(Float64, parts[1])
        ky = parse(Float64, parts[2])
        @test isapprox(kx, expected_k[idx][1]; atol=1e-9)
        @test isapprox(ky, expected_k[idx][2]; atol=1e-9)
        vals = parse.(Float64, parts[3:end])
        @test all(isfinite, vals)
        @test all(abs.(vals) .< 1e9)
    end
end
