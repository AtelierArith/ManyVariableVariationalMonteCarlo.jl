@testitem "outputs: struct/momentum content shape" begin
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

    # Expect L k-points for chain
    L = 6
    # zvo_struct.dat
    lines_struct = filter(l -> !isempty(l) && !startswith(l, "#"), readlines(joinpath(outdir, "zvo_struct.dat")))
    @test length(lines_struct) == L
    # zvo_momentum.dat
    lines_nk = filter(l -> !isempty(l) && !startswith(l, "#"), readlines(joinpath(outdir, "zvo_momentum.dat")))
    @test length(lines_nk) == L
end

@testitem "one-body green: diagonal equals snapshot occupancy" begin
    using ManyVariableVariationalMonteCarlo

    face = FaceDefinition()
    push_definition!(face, :model, "FermionHubbard")
    push_definition!(face, :lattice, "chain")
    push_definition!(face, :L, 5)
    push_definition!(face, :nelec, 4)
    push_definition!(face, :NVMCCalMode, 1)
    push_definition!(face, :NVMCSample, 10)

    config = SimulationConfig(face)
    layout = ParameterLayout(2, 0, 2, 0)

    sim = VMCSimulation(config, layout; T=ComplexF64)
    initialize_simulation!(sim)
    run_physics_calculation!(sim)

    outdir = mktempdir()
    output_results(sim, outdir)

    # Build expected occupancies from current state
    n = sim.vmc_state.n_sites
    nup = div(sim.vmc_state.n_electrons, 2)
    n_up = zeros(Int, n)
    n_dn = zeros(Int, n)
    for (k, pos) in enumerate(sim.vmc_state.electron_positions)
        if k <= nup
            n_up[pos] += 1
        else
            n_dn[pos] += 1
        end
    end

    # Parse zvo_cisajs.dat
    Gup = zeros(Float64, n, n)
    Gdn = zeros(Float64, n, n)
    for line in eachline(joinpath(outdir, "zvo_cisajs.dat"))
        if isempty(line) || startswith(line, "#"); continue; end
        parts = split(strip(line))
        # i s j t Re Im
        i = parse(Int, parts[1])
        s = parse(Int, parts[2])
        j = parse(Int, parts[3])
        t = parse(Int, parts[4])
        re = parse(Float64, parts[5])
        # im = parse(Float64, parts[6])
        if s == 1 && t == 1
            Gup[i,j] = re
        elseif s == 2 && t == 2
            Gdn[i,j] = re
        end
    end

    # Diagonals match occupancies
    for i in 1:n
        @test isapprox(Gup[i,i], n_up[i]; atol=1e-12)
        @test isapprox(Gdn[i,i], n_dn[i]; atol=1e-12)
    end
    # Off-diagonals are ~0
    for i in 1:n, j in 1:n
        if i != j
            @test isapprox(Gup[i,j], 0.0; atol=1e-12)
            @test isapprox(Gdn[i,j], 0.0; atol=1e-12)
        end
    end
end
