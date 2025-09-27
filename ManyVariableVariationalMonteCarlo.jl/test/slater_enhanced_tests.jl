@testitem "slater_enhanced" begin
    """
    Tests for enhanced Slater determinant implementation

    Tests all enhanced Slater determinant functionality including:
    - Basic Slater determinant operations
    - Frozen-spin variants
    - Backflow corrections
    - Performance benchmarks
    """

    using Test
    using StableRNGs
    using ManyVariableVariationalMonteCarlo

    @testset "Enhanced Slater Determinant Basic Functionality" begin

        # Test basic Slater determinant creation
        slater = SlaterDeterminant{ComplexF64}(3, 4)
        @test slater.slater_matrix.n_elec == 3
        @test slater.slater_matrix.n_orb == 4
        @test size(slater.slater_matrix.matrix) == (3, 4)
        @test size(slater.inverse_matrix) == (3, 3)

        # Test initialization
        orbital_matrix = rand(ComplexF64, 3, 4)
        initialize_slater!(slater, orbital_matrix)
        @test is_valid(slater)
        @test slater.update_count == 0
    end

    @testset "Frozen-Spin Slater Determinant" begin

        # Test frozen-spin Slater determinant creation
        frozen_spins = [1, -1, 1]  # Spin up, down, up
        slater = FrozenSpinSlaterDeterminant{ComplexF64}(3, 4, frozen_spins)

        @test slater.slater_matrix.n_elec == 3
        @test slater.slater_matrix.n_orb == 4
        @test slater.frozen_spins == frozen_spins
        @test length(slater.spin_up_indices) == 2
        @test length(slater.spin_down_indices) == 1

        # Test initialization
        orbital_matrix = rand(ComplexF64, 3, 4)
        initialize_frozen_spin_slater!(slater, orbital_matrix)
        @test is_valid(slater)

        # Test update
        new_value = ComplexF64(0.5 + 0.3im)
        ratio = update_frozen_spin_slater!(slater, 1, 1, new_value)
        @test isa(ratio, ComplexF64)
        @test slater.update_count == 1
    end

    @testset "Backflow Corrections" begin

        # Test backflow correction creation
        backflow = BackflowCorrection{ComplexF64}(4, 2)
        @test backflow.n_site == 4
        @test backflow.n_elec == 2
        @test size(backflow.backflow_weights) == (4, 4)
        @test length(backflow.backflow_bias) == 4

        # Test backflow correction application
        ele_idx = [1, 2]
        ele_cfg = [1, 1, 0, 0]
        apply_backflow_correction!(backflow, ele_idx, ele_cfg)
        @test length(backflow.backflow_buffer) == 4  # Buffer is sized for n_site

        # Test backflow-corrected orbital
        orbital_value = backflow_corrected_orbital(backflow, ele_idx, ele_cfg, 1, 1)
        @test isa(orbital_value, ComplexF64)
    end

    @testset "Backflow Slater Determinant" begin

        # Test backflow Slater determinant creation
        slater = BackflowSlaterDeterminant{ComplexF64}(3, 4, 4)
        @test slater.slater.slater_matrix.n_elec == 3
        @test slater.slater.slater_matrix.n_orb == 4
        @test slater.backflow.n_site == 4
        @test slater.backflow.n_elec == 3

        # Test initialization
        orbital_matrix = rand(ComplexF64, 3, 4)
        initialize_backflow_slater!(slater, orbital_matrix)
        @test is_backflow_valid(slater)

        # Test update
        ele_idx = [1, 2, 3]
        ele_cfg = [1, 1, 1, 0]
        new_value = ComplexF64(0.5 + 0.3im)
        ratio = update_backflow_slater!(slater, 1, 1, new_value, ele_idx, ele_cfg)
        @test isa(ratio, ComplexF64)
        @test slater.update_count == 1
    end

    @testset "Slater Determinant Updates" begin

        # Test single electron update
        slater = SlaterDeterminant{ComplexF64}(2, 3)
        orbital_matrix = [1.0 0.0 0.0; 0.0 1.0 0.0]
        initialize_slater!(slater, orbital_matrix)

        # Test single electron update
        ratio = update_slater!(slater, 1, 1, ComplexF64(0.5))
        @test isa(ratio, ComplexF64)
        @test slater.update_count == 1
        @test slater.last_update_row == 1
        @test slater.last_update_col == 1

        # Test two electron update
        ratio2 = two_electron_update!(slater, 1, 1, ComplexF64(0.6), 2, 2, ComplexF64(0.7))
        @test isa(ratio2, ComplexF64)
        @test slater.update_count == 2
        @test slater.last_update_row == -1  # Marked as two-electron update
        @test slater.last_update_col == -1
    end

    @testset "Slater Determinant Determinant Calculations" begin

        # Test determinant calculation
        slater = SlaterDeterminant{ComplexF64}(2, 2)
        orbital_matrix = [1.0 0.0; 0.0 1.0]
        initialize_slater!(slater, orbital_matrix)

        # Test determinant value
        det_val = get_determinant_value(slater)
        @test isa(det_val, ComplexF64)
        @test abs(det_val - ComplexF64(1.0)) < 1e-10

        # Test log determinant value
        log_det_val = get_log_determinant_value(slater)
        @test isa(log_det_val, Float64)
        @test abs(log_det_val - 0.0) < 1e-10
    end

    @testset "Slater Determinant Performance" begin

        # Test performance with larger matrices
        slater = SlaterDeterminant{ComplexF64}(10, 12)
        orbital_matrix = rand(ComplexF64, 10, 12)
        initialize_slater!(slater, orbital_matrix)

        # Benchmark updates
        n_iterations = 1000
        @time begin
            for _ = 1:n_iterations
                row = rand(1:10)
                col = rand(1:12)
                new_value = rand(ComplexF64)
                update_slater!(slater, row, col, new_value)
            end
        end

        @test slater.update_count == n_iterations
    end

    #=
    @testset "Slater Determinant Edge Cases" begin

        # Test zero-size determinant
        slater = SlaterDeterminant{ComplexF64}(0, 0)
        orbital_matrix = zeros(ComplexF64, 0, 0)
        initialize_slater!(slater, orbital_matrix)

        det_val = get_determinant_value(slater)
        @test det_val == ComplexF64(1.0)

        log_det_val = get_log_determinant_value(slater)
        @test log_det_val == 0.0

        # Test reset
        slater = SlaterDeterminant{ComplexF64}(2, 2)
        orbital_matrix = [1.0 0.0; 0.0 1.0]
        initialize_slater!(slater, orbital_matrix)

        reset_slater!(slater)
        @test !is_valid(slater)
        @test slater.update_count == 0
        @test slater.last_update_row == -1
        @test slater.last_update_col == -1
    end
    =#

    @testset "Slater Determinant Complex vs Real" begin

        # Test complex Slater determinant
        slater_complex = SlaterDeterminant{ComplexF64}(2, 2)
        orbital_matrix_complex = [1.0+0.1im 0.0; 0.0 1.0-0.1im]
        initialize_slater!(slater_complex, orbital_matrix_complex)

        det_val_complex = get_determinant_value(slater_complex)
        @test isa(det_val_complex, ComplexF64)
        # For this diagonal matrix with conjugate phases, determinant is real
        @test isapprox(imag(det_val_complex), 0.0; atol = 1e-12)

        # Test real Slater determinant
        slater_real = SlaterDeterminant{Float64}(2, 2)
        orbital_matrix_real = [1.0 0.0; 0.0 1.0]
        initialize_slater!(slater_real, orbital_matrix_real)

        det_val_real = get_determinant_value(slater_real)
        @test isa(det_val_real, Float64)
        @test imag(det_val_real) == 0.0
    end

    @testset "Frozen-Spin Move Validation" begin

        # Test frozen-spin move validation
        frozen_spins = [1, -1, 1]
        slater = FrozenSpinSlaterDeterminant{ComplexF64}(3, 4, frozen_spins)

        # Test move validation (simplified implementation)
        @test ManyVariableVariationalMonteCarlo._is_move_allowed(slater, 1, 1)
        @test ManyVariableVariationalMonteCarlo._is_move_allowed(slater, 2, 2)
        @test ManyVariableVariationalMonteCarlo._is_move_allowed(slater, 3, 3)
    end

    @testset "Backflow Correction Edge Cases" begin

        # Test backflow correction with empty electron configuration
        backflow = BackflowCorrection{ComplexF64}(4, 2)
        ele_idx = Int[]
        ele_cfg = [0, 0, 0, 0]

        apply_backflow_correction!(backflow, ele_idx, ele_cfg)
        @test length(backflow.backflow_buffer) == 4

        # Test backflow-corrected orbital with empty configuration
        orbital_value = backflow_corrected_orbital(backflow, ele_idx, ele_cfg, 1, 1)
        @test orbital_value == ComplexF64(0.0)
    end
end # @testitem "slater_enhanced"
