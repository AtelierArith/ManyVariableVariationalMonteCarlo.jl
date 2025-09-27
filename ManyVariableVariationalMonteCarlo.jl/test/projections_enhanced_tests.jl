@testitem "projections_enhanced" begin
    """
    Tests for enhanced quantum projections implementation

    Tests all enhanced quantum projection functionality including:
    - Basic quantum projections
    - Point group projections with symmetry operations
    - Advanced projection types (time reversal, particle-hole)
    - Performance benchmarks
    """

    using Test
    using StableRNGs
    using ManyVariableVariationalMonteCarlo

    @testset "Enhanced Quantum Projections Basic Functionality" begin

        # Test basic quantum projection creation
        qp = QuantumProjection{ComplexF64}(4, 2)
        @test qp.n_site == 4
        @test qp.n_elec == 2
        @test qp.n_spin == 2
        @test isempty(qp.spin_projections)
        @test isempty(qp.momentum_projections)
        @test isempty(qp.particle_number_projections)
        @test isempty(qp.parity_projections)

        # Test projection addition
        add_spin_projection!(qp, ComplexF64(0.0), ComplexF64(1.0))
        add_particle_number_projection!(qp, 2, ComplexF64(1.0))
        add_parity_projection!(qp, 1, ComplexF64(1.0))

        @test length(qp.spin_projections) == 1
        @test length(qp.particle_number_projections) == 1
        @test length(qp.parity_projections) == 1
    end

    @testset "Point Group Projections" begin

        # Test point group projection creation
        pgp = PointGroupProjection{ComplexF64}(4, 2)
        @test pgp.n_site == 4
        @test pgp.n_elec == 2
        @test isempty(pgp.symmetry_operations)

        # Test symmetry operation addition
        identity_matrix = Matrix{ComplexF64}(I, 4, 4)
        add_symmetry_operation!(pgp, "Identity", identity_matrix, ComplexF64(1.0))
        @test length(pgp.symmetry_operations) == 1
        @test pgp.symmetry_operations[1].operation_type == "Identity"
        @test pgp.symmetry_operations[1].phase_factor == ComplexF64(1.0)

        # Test point group ratio calculation
        ele_idx = [1, 2]
        ele_cfg = [1, 1, 0, 0]
        ele_num = [1, 1, 0, 0]

        ratio = calculate_point_group_ratio(pgp, ele_idx, ele_cfg, ele_num)
        @test isa(ratio, ComplexF64)
        @test ratio != zero(ComplexF64)
    end

    @testset "Time Reversal Projections" begin

        # Test time reversal projection creation
        trp = TimeReversalProjection{ComplexF64}(4, 2, ComplexF64(0.5))
        @test trp.n_site == 4
        @test trp.n_elec == 2
        @test trp.time_reversal_phase == ComplexF64(0.5)
        @test trp.is_active

        # Test time reversal ratio calculation
        ele_idx = [1, 2]
        ele_cfg = [1, 1, 0, 0]
        ele_num = [1, 1, 0, 0]

        ratio = calculate_time_reversal_ratio(trp, ele_idx, ele_cfg, ele_num)
        @test isa(ratio, ComplexF64)
        @test ratio == ComplexF64(0.5)

        # Test inactive time reversal projection
        trp.is_active = false
        ratio_inactive = calculate_time_reversal_ratio(trp, ele_idx, ele_cfg, ele_num)
        @test ratio_inactive == ComplexF64(1.0)
    end

    @testset "Particle-Hole Projections" begin

        # Test particle-hole projection creation
        php = ParticleHoleProjection{ComplexF64}(4, 2, ComplexF64(0.3))
        @test php.n_site == 4
        @test php.n_elec == 2
        @test php.particle_hole_phase == ComplexF64(0.3)
        @test php.is_active

        # Test particle-hole ratio calculation
        ele_idx = [1, 2]
        ele_cfg = [1, 1, 0, 0]
        ele_num = [1, 1, 0, 0]

        ratio = calculate_particle_hole_ratio(php, ele_idx, ele_cfg, ele_num)
        @test isa(ratio, ComplexF64)
        @test ratio == ComplexF64(0.3)

        # Test inactive particle-hole projection
        php.is_active = false
        ratio_inactive = calculate_particle_hole_ratio(php, ele_idx, ele_cfg, ele_num)
        @test ratio_inactive == ComplexF64(1.0)
    end

    @testset "Advanced Quantum Projections" begin

        # Test advanced quantum projection creation
        aqp = AdvancedQuantumProjection{ComplexF64}(4, 2)
        @test aqp.n_site == 4
        @test aqp.n_elec == 2
        @test aqp.n_spin == 2
        @test isa(aqp.point_group_projection, PointGroupProjection{ComplexF64})
        @test isa(aqp.time_reversal_projection, TimeReversalProjection{ComplexF64})
        @test isa(aqp.particle_hole_projection, ParticleHoleProjection{ComplexF64})

        # Test advanced projection ratio calculation
        ele_idx = [1, 2]
        ele_cfg = [1, 1, 0, 0]
        ele_num = [1, 1, 0, 0]

        ratio = calculate_advanced_projection_ratio(aqp, ele_idx, ele_cfg, ele_num)
        @test isa(ratio, ComplexF64)
        @test_broken ratio != zero(ComplexF64)
        @test aqp.total_projections == 1
    end

    @testset "Cubic Symmetry Setup" begin

        # Test cubic symmetry setup
        aqp = AdvancedQuantumProjection{ComplexF64}(4, 2)
        setup_cubic_symmetry!(aqp)

        # Check that symmetry operations were added
        @test length(aqp.point_group_projection.symmetry_operations) >= 1

        # Check for identity operation
        identity_ops = filter(
            op -> op.operation_type == "Identity",
            aqp.point_group_projection.symmetry_operations,
        )
        @test length(identity_ops) == 1

        # Test symmetry operation application
        ele_idx = [1, 2]
        ele_cfg = [1, 1, 0, 0]
        ele_num = [1, 1, 0, 0]

        ratio = calculate_point_group_ratio(
            aqp.point_group_projection,
            ele_idx,
            ele_cfg,
            ele_num,
        )
        @test isa(ratio, ComplexF64)
        @test ratio != zero(ComplexF64)
    end

    @testset "Gauss-Legendre Quadrature" begin

        # Test Gauss-Legendre quadrature
        points, weights = gauss_legendre_quadrature(5, ComplexF64(-1.0), ComplexF64(1.0))
        @test length(points) == 5
        @test length(weights) == 5
        @test all(isa.(points, ComplexF64))
        @test all(isa.(weights, ComplexF64))

        # Test quadrature properties
        @test all(-1.0 .<= real.(points) .<= 1.0)
        @test all(real.(weights) .> 0.0)

        # Test integration of constant function
        integral = sum(weights)
        @test abs(real(integral) - 2.0) < 1e-10  # Should integrate to 2.0 over [-1, 1]
    end

    @testset "Continuous Projections" begin

        # Test continuous projection setup
        proj = ProjectionOperator{ComplexF64}(
            MOMENTUM_PROJECTION,
            ComplexF64(0.0),
            ComplexF64(1.0),
        )
        setup_continuous_projection!(proj, 10, ComplexF64(-1.0), ComplexF64(1.0))

        @test length(proj.integration_points) == 10
        @test length(proj.integration_weights) == 10

        # Test continuous projection ratio calculation
        ele_idx = [1, 2]
        ele_cfg = [1, 1, 0, 0]
        ele_num = [1, 1, 0, 0]

        ratio = calculate_continuous_projection_ratio(proj, ele_idx, ele_cfg, ele_num)
        @test isa(ratio, ComplexF64)
        @test ratio != zero(ComplexF64)
    end

    @testset "Projection Performance" begin

        # Test performance with larger systems
        aqp = AdvancedQuantumProjection{ComplexF64}(10, 5)
        setup_cubic_symmetry!(aqp)

        ele_idx = collect(1:5)
        ele_cfg = zeros(Int, 10)
        ele_cfg[1:5] .= 1
        ele_num = copy(ele_cfg)

        # Benchmark advanced projection calculations
        n_iterations = 1000
        @time begin
            for _ = 1:n_iterations
                calculate_advanced_projection_ratio(aqp, ele_idx, ele_cfg, ele_num)
            end
        end

        @test aqp.total_projections == n_iterations
    end

    @testset "Projection Edge Cases" begin

        # Test empty projection
        qp = QuantumProjection{ComplexF64}(4, 2)
        ele_idx = [1, 2]
        ele_cfg = [1, 1, 0, 0]
        ele_num = [1, 1, 0, 0]

        ratio = calculate_projection_ratio(qp, ele_idx, ele_cfg, ele_num)
        @test ratio == ComplexF64(1.0)  # Should return 1.0 for empty projection

        # Test projection with no active operators
        add_spin_projection!(qp, ComplexF64(0.0), ComplexF64(1.0))
        qp.spin_projections[1].is_active = false

        ratio_inactive = calculate_projection_ratio(qp, ele_idx, ele_cfg, ele_num)
        @test ratio_inactive == ComplexF64(1.0)
    end

    @testset "Projection Complex vs Real" begin

        # Test complex projections
        qp_complex = QuantumProjection{ComplexF64}(4, 2)
        add_spin_projection!(qp_complex, ComplexF64(0.0), ComplexF64(1.0 + 0.1im))

        ele_idx = [1, 2]
        ele_cfg = [1, 1, 0, 0]
        ele_num = [1, 1, 0, 0]

        ratio_complex = calculate_projection_ratio(qp_complex, ele_idx, ele_cfg, ele_num)
        @test isa(ratio_complex, ComplexF64)
        @test imag(ratio_complex) != 0.0

        # Test real projections
        qp_real = QuantumProjection{Float64}(4, 2)
        add_spin_projection!(qp_real, 0.0, 1.0)

        ratio_real = calculate_projection_ratio(qp_real, ele_idx, ele_cfg, ele_num)
        @test isa(ratio_real, Float64)
        @test imag(ratio_real) == 0.0
    end

    @testset "Symmetry Operation Edge Cases" begin

        # Test symmetry operation with wrong matrix size
        pgp = PointGroupProjection{ComplexF64}(4, 2)
        wrong_matrix = Matrix{ComplexF64}(I, 3, 3)

        @test_throws ArgumentError add_symmetry_operation!(
            pgp,
            "Wrong",
            wrong_matrix,
            ComplexF64(1.0),
        )

        # Test symmetry operation with correct matrix size
        correct_matrix = Matrix{ComplexF64}(I, 4, 4)
        add_symmetry_operation!(pgp, "Correct", correct_matrix, ComplexF64(1.0))
        @test length(pgp.symmetry_operations) == 1
    end

    @testset "Projection Statistics" begin

        # Test projection statistics tracking
        qp = QuantumProjection{ComplexF64}(4, 2)
        add_spin_projection!(qp, ComplexF64(0.0), ComplexF64(1.0))

        ele_idx = [1, 2]
        ele_cfg = [1, 1, 0, 0]
        ele_num = [1, 1, 0, 0]

        # Perform multiple projections
        for _ = 1:100
            calculate_projection_ratio(qp, ele_idx, ele_cfg, ele_num)
        end

        @test qp.total_projections == 100
    end
end # @testitem "projections_enhanced"
