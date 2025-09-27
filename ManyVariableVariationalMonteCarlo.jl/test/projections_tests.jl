"""
Tests for Quantum Projections implementation (Phase 2: Mathematical Foundation)
"""


@testitem "Projection Basic Operations" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    using Test
    using ManyVariableVariationalMonteCarlo: QuantumProjection, ProjectionType,
                                           add_spin_projection!, add_particle_number_projection!,
                                           add_parity_projection!, calculate_projection_ratio

    # Test basic projection creation
    n_site = 4
    n_elec = 2
    qp = QuantumProjection{ComplexF64}(n_site, n_elec)

    @test qp.n_site == n_site
    @test qp.n_elec == n_elec
    @test qp.n_spin == 2

    # Test adding projections
    add_spin_projection!(qp, ComplexF64(0.0), ComplexF64(1.0))
    add_particle_number_projection!(qp, n_elec, ComplexF64(1.0))
    add_parity_projection!(qp, 1, ComplexF64(1.0))

    @test length(qp.spin_projections) == 1
    @test length(qp.particle_number_projections) == 1
    @test length(qp.parity_projections) == 1

    # Test electron configuration setup
    ele_idx = [1, 2]
    ele_cfg = zeros(Int, n_site)
    ele_cfg[1:2] .= 1
    ele_num = copy(ele_cfg)

    # Test projection ratio calculation
    ratio = calculate_projection_ratio(qp, ele_idx, ele_cfg, ele_num)
    @test isa(ratio, ComplexF64)
    @test qp.total_projections == 1
end

@testitem "Gauss-Legendre Quadrature" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    using Test
    using ManyVariableVariationalMonteCarlo: gauss_legendre_quadrature

    # Test basic quadrature
    n = 5
    a, b = -1.0, 1.0
    points, weights = gauss_legendre_quadrature(n, a, b)

    @test length(points) == n
    @test length(weights) == n
    @test all(isa.(points, Float64))
    @test all(isa.(weights, Float64))

    # Test that points are in correct range
    @test all(a .<= points .<= b)

    # Test that weights are positive
    @test all(weights .> 0)

    # Test integration of constant function
    integral = sum(weights)
    @test isapprox(integral, b - a, rtol=1e-10)

    # Test integration of x^2
    integral_x2 = sum(weights .* points.^2)
    expected = (b^3 - a^3) / 3
    @test isapprox(integral_x2, expected, rtol=1e-10)
end

@testitem "Continuous Projections" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    using Test
    using ManyVariableVariationalMonteCarlo: QuantumProjection, ProjectionOperator, MOMENTUM_PROJECTION,
                                           setup_continuous_projection!, calculate_continuous_projection_ratio

    # Test continuous projection setup
    qp = QuantumProjection{ComplexF64}(4, 2)
    proj = add_momentum_projection!(qp, [ComplexF64(0.0)], ComplexF64(1.0))

    # Setup continuous projection
    setup_continuous_projection!(proj, 10, ComplexF64(-π), ComplexF64(π))

    @test length(proj.integration_points) == 10
    @test length(proj.integration_weights) == 10
    @test all(isa.(proj.integration_points, ComplexF64))
    @test all(isa.(proj.integration_weights, ComplexF64))

    # Test continuous projection calculation
    ele_idx = [1, 2]
    ele_cfg = zeros(Int, 4)
    ele_cfg[1:2] .= 1
    ele_num = copy(ele_cfg)

    ratio = calculate_continuous_projection_ratio(proj, ele_idx, ele_cfg, ele_num)
    @test isa(ratio, ComplexF64)
end

@testitem "Projection Edge Cases" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    using Test
    using ManyVariableVariationalMonteCarlo: QuantumProjection, add_spin_projection!,
                                           calculate_projection_ratio

    # Test with empty projections
    qp = QuantumProjection{ComplexF64}(3, 1)

    ele_idx = [1]
    ele_cfg = zeros(Int, 3)
    ele_cfg[1] = 1
    ele_num = copy(ele_cfg)

    # No projections should return 1.0
    ratio = calculate_projection_ratio(qp, ele_idx, ele_cfg, ele_num)
    @test ratio == ComplexF64(1.0)

    # Test with multiple projections
    add_spin_projection!(qp, ComplexF64(0.5), ComplexF64(2.0))
    add_spin_projection!(qp, ComplexF64(-0.5), ComplexF64(1.5))

    @test length(qp.spin_projections) == 2

    ratio = calculate_projection_ratio(qp, ele_idx, ele_cfg, ele_num)
    @test isa(ratio, ComplexF64)
end

@testitem "Projection Performance" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    using Test
    using ManyVariableVariationalMonteCarlo: QuantumProjection, add_spin_projection!,
                                           add_particle_number_projection!, calculate_projection_ratio

    # Test with larger system
    n_site = 10
    n_elec = 5
    qp = QuantumProjection{ComplexF64}(n_site, n_elec)

    # Add multiple projections
    for i in 1:5
        add_spin_projection!(qp, ComplexF64(i-3), ComplexF64(1.0))
    end
    add_particle_number_projection!(qp, n_elec, ComplexF64(1.0))

    ele_idx = collect(1:n_elec)
    ele_cfg = zeros(Int, n_site)
    ele_cfg[1:n_elec] .= 1
    ele_num = copy(ele_cfg)

    # Test multiple calculations
    ratios = ComplexF64[]
    for _ in 1:100
        ratio = calculate_projection_ratio(qp, ele_idx, ele_cfg, ele_num)
        push!(ratios, ratio)
    end

    @test length(ratios) == 100
    @test all(isa.(ratios, ComplexF64))
    @test qp.total_projections == 100
end

@testitem "Projection Benchmark" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    using Test
    using ManyVariableVariationalMonteCarlo: benchmark_projections

    # Test that benchmark runs without error
    @test_nowarn benchmark_projections(5, 3, 10)
end