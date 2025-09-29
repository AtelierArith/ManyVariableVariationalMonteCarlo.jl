@testitem "Load mVMC StdFace.def (Heisenberg chain)" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    # Locate sample StdFace.def from the repo
    pkgroot = pkgdir(ManyVariableVariationalMonteCarlo)
    stdface_path = joinpath(pkgroot, "mVMC", "samples", "Standard", "Spin", "HeisenbergChain", "StdFace.def")
    @test isfile(stdface_path)

    # Load face and map to SimulationConfig
    face = load_face_definition(stdface_path)
    @test facevalue(face, :L, Int) == 16
    @test facevalue(face, :J, Float64) == 1.0
    @test facevalue(face, :model, String) == "Spin"
    @test facevalue(face, :lattice, String) == "chain"

    # Provide tiny sampling defaults to keep the test fast
    if !haskey(face, :NVMCSample)
        push_definition!(face, :NVMCSample, 8)
    end
    if !haskey(face, :NVMCWarmUp)
        push_definition!(face, :NVMCWarmUp, 4)
    end
    push_definition!(face, :NVMCCalMode, 1)  # Physics calculation

    simcfg = SimulationConfig(face)
    @test simcfg.model == :Spin
    @test simcfg.lattice == :chain
    @test simcfg.nsites == 16

    # Build and initialize a minimal simulation
    layout = ParameterLayout(0, 0, 0, 0)
    sim = VMCSimulation(simcfg, layout)
    initialize_simulation!(sim)

    @test sim.vmc_state !== nothing
    @test sim.vmc_state.hamiltonian !== nothing
    @test length(sim.vmc_state.hamiltonian.hund_terms) > 0

    # Run a tiny physics calculation and write results to a temp dir
    sim.mode = PHYSICS_CALCULATION
    run_physics_calculation!(sim)
    tmp = mktempdir()
    output_results(sim, tmp)
    @test isfile(joinpath(tmp, "zvo_result.dat"))
end

@testitem "Heisenberg Hamiltonian supports square lattice" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    # 2x2 square
    n_sites = 4
    J = 1.0
    ham = create_heisenberg_hamiltonian(n_sites, J; lattice_type = :square)
    @test ham.n_sites == n_sites
    @test length(ham.hund_terms) > 0
end

@testitem "Heisenberg via StdFace geometry (triangular, honeycomb, ladder)" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    # Triangular lattice
    geom_tri = create_triangular_lattice(3, 3)
    ham_tri = create_heisenberg_hamiltonian(geom_tri, 1.0)
    @test ham_tri.n_sites == geom_tri.n_sites_total
    @test length(ham_tri.hund_terms) > 0

    # Honeycomb lattice
    geom_honey = create_honeycomb_lattice(2, 2)
    ham_honey = create_heisenberg_hamiltonian(geom_honey, 1.0)
    @test ham_honey.n_sites == geom_honey.n_sites_total
    @test length(ham_honey.hund_terms) > 0

    # Ladder lattice (2-leg)
    geom_lad = create_ladder_lattice(3, 2)
    ham_lad = create_heisenberg_hamiltonian(geom_lad, 1.0)
    @test ham_lad.n_sites == geom_lad.n_sites_total
    @test length(ham_lad.hund_terms) > 0

    # Kagome lattice
    geom_kag = create_kagome_lattice(2, 2)
    ham_kag = create_heisenberg_hamiltonian(geom_kag, 1.0)
    @test ham_kag.n_sites == geom_kag.n_sites_total
    @test length(ham_kag.hund_terms) > 0
end

@testitem "Heisenberg n_sites API supports triangular/honeycomb/ladder" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    # Triangular 3x3
    ham_tri = create_heisenberg_hamiltonian(9, 1.0; lattice_type = :triangular)
    @test ham_tri.n_sites == 9
    @test length(ham_tri.hund_terms) > 0

    # Honeycomb 2x2 unit cells => 8 sites
    ham_honey = create_heisenberg_hamiltonian(8, 1.0; lattice_type = :honeycomb)
    @test ham_honey.n_sites == 8
    @test length(ham_honey.hund_terms) > 0

    # Ladder 2-leg with L=5 => 10 sites
    ham_lad = create_heisenberg_hamiltonian(10, 1.0; lattice_type = :ladder)
    @test ham_lad.n_sites == 10
    @test length(ham_lad.hund_terms) > 0

    # Kagome 2x2 unit cells => 12 sites
    ham_kag = create_heisenberg_hamiltonian(12, 1.0; lattice_type = :kagome)
    @test ham_kag.n_sites == 12
    @test length(ham_kag.hund_terms) > 0
end
