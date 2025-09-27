@testitem "StdFace.def parsing" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    package_root = normpath(joinpath(@__DIR__, ".."))
    face_path = joinpath(repo_root, "mVMC", "samples", "Standard", "Hubbard", "square", "StdFace.def")
    face = load_face_definition(face_path)
    cfg = SimulationConfig(face; root=package_root)
    @test facevalue(face, :W, Int) == 4
    @test facevalue(face, :L, Int) == 2
    @test facevalue(face, :U, Float64) == 4.0
    @test cfg.nsites == 8
    @test cfg.nsublat == 4
    @test cfg.nsite_sub == 2
    @test cfg.model == :FermionHubbard
    @test cfg.lattice == :Tetragonal
end
@testitem "Green function loader" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    using Base: Set
    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    green_path = joinpath(repo_root, "mVMC", "samples", "Standard", "Hubbard", "square", "UHF", "initial.def")
    table = read_initial_green(green_path)
    @test length(table) == 128
    @test table[1].value ≈ ComplexF64(0.49099999999999997, 0)
    unique_indices = Set(entry.bra for entry in table)
    @test !isempty(unique_indices)
end
@testitem "Parameter initialisation replicates C heuristics" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    layout = ParameterLayout(3, 2, 4, 2)
    mask = ParameterMask(layout; default=true)
    flags = ParameterFlags(false, true)
    params = ParameterSet(layout)
    rng = StableRNG(1)
    qp_values = fill(1.0 + 0im, layout.nopttrans)
    initialize_parameters!(params, layout, mask, flags; rng=rng, para_qp_opttrans=qp_values, rbm_scale=4)
    @test all(iszero, params.proj)
    @test params.opttrans == qp_values
    @test maximum(abs, params.slater) <= ManyVariableVariationalMonteCarlo.AMP_MAX + 1e-12
    rng_replay = StableRNG(1)
    expected_rbm = similar(params.rbm)
    for i in eachindex(expected_rbm)
        expected_rbm[i] = 0.01 * (rand(rng_replay) - 0.5) / 4
    end
    @test params.rbm == expected_rbm
end
@testitem "Definition parsers" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    root = normpath(joinpath(@__DIR__, "..", "..", "mVMC", "test", "python", "data", "UHF_InterAll_Exchange"))
    namelist = load_namelist(joinpath(root, "namelist_all.def"))
    @test length(namelist) == 6
    @test namelist[1].key == :ModPara
    @test namelist[1].path == "modpara.def"
    @test namelist[end].key == :TwoBodyG
    transfers = read_transfer_table(joinpath(root, "trans.def"))
    @test length(transfers) == 16
    first_transfer = transfers[1]
    @test first_transfer.amplitude ≈ 1.0 + 0im
    coulomb = read_coulomb_intra(joinpath(root, "coulombintra.def"))
    @test length(coulomb) == 4
    @test coulomb[1].value ≈ 4.0
    inter = read_interall_table(joinpath(root, "interall.def"))
    @test length(inter) == 20
    @test inter[1].indices == (0, 0, 0, 0, 0, 1, 0, 1)
    green = read_greenone_indices(joinpath(root, "greenone.def"))
    @test length(green) == 8
    @test green[1].bra == (0, 0)
    @test green[1].ket == (0, 0)
end
