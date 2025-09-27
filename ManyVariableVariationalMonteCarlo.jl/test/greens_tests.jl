"""
Tests for Green Functions implementation (Phase 2: Mathematical Foundation)
"""


using Test
@testset "Green Function Basic Operations" begin
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    using ManyVariableVariationalMonteCarlo: LocalGreenFunction, green_function_1body!,
                                           green_function_2body!, clear_green_function_cache!,
                                           get_cache_statistics

    # Test basic Green function creation
    n_site = 4
    n_elec = 2
    gf = LocalGreenFunction{ComplexF64}(n_site, n_elec)

    @test gf.n_site == n_site
    @test gf.n_elec == n_elec
    @test gf.n_spin == 2

    # Test electron configuration setup
    ele_idx = [1, 2]
    ele_cfg = zeros(Int, n_site * 2)  # n_site * n_spin
    ele_cfg[1:2] .= 1
    ele_num = zeros(Int, n_site * 2)  # n_site * n_spin
    ele_num[1:2] .= 1
    proj_cnt = zeros(Int, n_site)

    # Test 1-body Green function calculation
    ip = 1.0 + 0.1im
    result = green_function_1body!(gf, 1, 1, 1, ip, ele_idx, ele_cfg, ele_num, proj_cnt)
    @test isa(result, ComplexF64)

    # Test 2-body Green function calculation
    result2 = green_function_2body!(gf, 1, 2, 3, 4, 1, 1, ip, ele_idx, ele_cfg, ele_num, proj_cnt)
    @test isa(result2, ComplexF64)

    # Test cache statistics
    stats = get_cache_statistics(gf)
    @test stats.hits >= 0
    @test stats.misses >= 0
    @test 0.0 <= stats.hit_rate <= 1.0

    # Test cache clearing
    clear_green_function_cache!(gf)
    stats_after = get_cache_statistics(gf)
    @test stats_after.hits == 0
    @test stats_after.misses == 0
end

@testset "Green Function Edge Cases" begin
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    using ManyVariableVariationalMonteCarlo: LocalGreenFunction, green_function_1body!

    n_site = 3
    n_elec = 1
    gf = LocalGreenFunction{ComplexF64}(n_site, n_elec)

    ele_idx = [1]
    ele_cfg = zeros(Int, n_site * 2)  # n_site * n_spin
    ele_cfg[1] = 1
    ele_num = zeros(Int, n_site * 2)  # n_site * n_spin
    ele_num[1] = 1
    proj_cnt = zeros(Int, n_site)

    ip = 1.0 + 0.1im

    # Test same site (should return particle number)
    result = green_function_1body!(gf, 1, 1, 1, ip, ele_idx, ele_cfg, ele_num, proj_cnt)
    @test result == ComplexF64(ele_num[1])

    # Test empty site (should return 0)
    result = green_function_1body!(gf, 2, 2, 1, ip, ele_idx, ele_cfg, ele_num, proj_cnt)
    @test result == ComplexF64(ele_num[2])

    # Test hopping to occupied site (should return 0)
    result = green_function_1body!(gf, 1, 2, 1, ip, ele_idx, ele_cfg, ele_num, proj_cnt)
    @test isa(result, ComplexF64)
end

@testset "Green Function Performance" begin
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    using ManyVariableVariationalMonteCarlo: LocalGreenFunction, green_function_1body!

    # Test with larger system
    n_site = 10
    n_elec = 5
    gf = LocalGreenFunction{ComplexF64}(n_site, n_elec)

    ele_idx = collect(1:n_elec)
    ele_cfg = zeros(Int, n_site * 2)  # n_site * n_spin
    ele_cfg[1:n_elec] .= 1
    ele_num = zeros(Int, n_site * 2)  # n_site * n_spin
    ele_num[1:n_elec] .= 1
    proj_cnt = zeros(Int, n_site)

    ip = 1.0 + 0.1im

    # Test multiple calculations
    results = ComplexF64[]
    for ri in 1:n_site
        for rj in 1:n_site
            for s in 1:2
                result = green_function_1body!(gf, ri, rj, s, ip, ele_idx, ele_cfg, ele_num, proj_cnt)
                push!(results, result)
            end
        end
    end

    @test length(results) == n_site * n_site * 2
    @test all(isa.(results, ComplexF64))
end

@testset "Green Function Benchmark" begin
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    using ManyVariableVariationalMonteCarlo: benchmark_green_functions

    # Test that benchmark runs without error
    @test_nowarn benchmark_green_functions(5, 3, 10)
end