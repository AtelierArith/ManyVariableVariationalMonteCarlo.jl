@testitem "rbm_enhanced" begin
    """
    Tests for enhanced RBM implementation

    Tests all enhanced RBM functionality including:
    - Basic RBM operations
    - Efficient gradient calculations
    - Variational parameter management
    - RBM ensembles
    - Performance benchmarks
    """

    using Test
    using StableRNGs
    using ManyVariableVariationalMonteCarlo

    @testset "Enhanced RBM Basic Functionality" begin
        using ManyVariableVariationalMonteCarlo

        # Test RBM creation
        rbm = RBMNetwork{ComplexF64}(4, 6, 2)
        @test rbm.n_visible == 4
        @test rbm.n_hidden == 6
        @test rbm.n_phys_layer == 2
        @test size(rbm.weights) == (6, 4)
        @test length(rbm.visible_bias) == 4
        @test length(rbm.hidden_bias) == 6
        @test size(rbm.phys_weights) == (6, 2)
        @test !rbm.is_initialized

        # Test initialization
        initialize_rbm!(rbm, scale = 0.01)
        @test rbm.is_initialized
        @test rbm.is_complex
    end

    @testset "RBM Weight Calculations" begin
        using ManyVariableVariationalMonteCarlo

        # Test RBM weight calculation
        rbm = RBMNetwork{ComplexF64}(3, 4, 2)
        initialize_rbm!(rbm, scale = 0.01)

        visible_state = [1, 0, 1]
        hidden_state = [1, 0, 1, 0]
        phys_state = [0.5, 0.3]

        # Test basic weight calculation
        weight = rbm_weight(rbm, visible_state, hidden_state)
        @test isa(weight, ComplexF64)
        @test weight != zero(ComplexF64)

        # Test log weight calculation
        log_weight = log_rbm_weight(rbm, visible_state, hidden_state)
        @test isa(log_weight, ComplexF64)
        @test abs(exp(log_weight) - weight) < 1e-10

        # Test physical layer weight calculation
        weight_phys = rbm_weight_phys(rbm, visible_state, hidden_state, phys_state)
        @test isa(weight_phys, ComplexF64)
        @test weight_phys != zero(ComplexF64)
    end

    @testset "RBM Gradient Calculations" begin
        using ManyVariableVariationalMonteCarlo

        # Test RBM gradient calculation
        rbm = RBMNetwork{ComplexF64}(3, 4, 2)
        initialize_rbm!(rbm, scale = 0.01)

        visible_state = [1, 0, 1]
        hidden_state = [1, 0, 1, 0]
        phys_state = [0.5, 0.3]

        # Test basic gradient calculation
        grad = rbm_gradient(rbm, visible_state, hidden_state)
        @test isa(grad.weights, Matrix{ComplexF64})
        @test isa(grad.visible_bias, Vector{ComplexF64})
        @test isa(grad.hidden_bias, Vector{ComplexF64})
        @test size(grad.weights) == (4, 3)
        @test length(grad.visible_bias) == 3
        @test length(grad.hidden_bias) == 4

        # Test physical layer gradient calculation
        grad_phys = rbm_gradient_phys(rbm, visible_state, hidden_state, phys_state)
        @test isa(grad_phys.weights, Matrix{ComplexF64})
        @test isa(grad_phys.phys_weights, Matrix{ComplexF64})
        @test size(grad_phys.phys_weights) == (4, 2)
    end

    @testset "Efficient RBM Gradient Calculations" begin
        using ManyVariableVariationalMonteCarlo

        # Test efficient gradient calculation
        rbm = RBMNetwork{ComplexF64}(3, 4, 2)
        initialize_rbm!(rbm, scale = 0.01)

        grad = RBMGradient{ComplexF64}(3, 4, 2)
        visible_state = [1, 0, 1]
        hidden_state = [1, 0, 1, 0]
        phys_state = [0.5, 0.3]

        # Test efficient gradient computation
        compute_efficient_gradient!(grad, rbm, visible_state, hidden_state, phys_state)
        @test grad.gradient_count == 1
        @test grad.gradient_norm > 0.0
        @test grad.max_gradient > 0.0

        # Test gradient statistics
        @test isa(grad.gradient_norm, Float64)
        @test isa(grad.max_gradient, Float64)
        @test isa(grad.gradient_count, Int)
    end

    @testset "RBM Parameter Management" begin
        using ManyVariableVariationalMonteCarlo

        # Test RBM parameter management
        rbm = RBMNetwork{ComplexF64}(3, 4, 2)
        initialize_rbm!(rbm, scale = 0.01)

        # Test parameter extraction
        params = get_rbm_parameters(rbm)
        @test isa(params, Vector{ComplexF64})
        @test length(params) == rbm_parameter_count(rbm)

        # Test parameter setting
        new_params = rand(ComplexF64, length(params))
        set_rbm_parameters!(rbm, new_params)
        updated_params = get_rbm_parameters(rbm)
        @test updated_params == new_params

        # Test parameter count
        n_params = rbm_parameter_count(rbm)
        expected_count = 4 * 3 + 3 + 4 + 4 * 2  # weights + visible_bias + hidden_bias + phys_weights
        @test n_params == expected_count
    end

    @testset "Variational RBM Management" begin
        using ManyVariableVariationalMonteCarlo

        # Test variational RBM creation
        vrbm = VariationalRBM{ComplexF64}(3, 4, 2)
        @test vrbm.rbm.n_visible == 3
        @test vrbm.rbm.n_hidden == 4
        @test vrbm.rbm.n_phys_layer == 2
        @test length(vrbm.parameter_names) > 0
        @test length(vrbm.parameter_bounds) > 0
        @test length(vrbm.parameter_scales) > 0

        # Test parameter bounds setting
        set_parameter_bounds!(vrbm, "param_1", ComplexF64(-1.0), ComplexF64(1.0))
        @test vrbm.parameter_bounds[1] == (ComplexF64(-1.0), ComplexF64(1.0))

        # Test parameter scale setting
        set_parameter_scale!(vrbm, "param_1", 2.0)
        @test vrbm.parameter_scales[1] == 2.0

        # Test parameter value getting and setting
        initialize_rbm!(vrbm.rbm, scale = 0.01)
        param_value = get_parameter_value(vrbm, "param_1")
        @test isa(param_value, ComplexF64)

        set_parameter_value!(vrbm, "param_1", ComplexF64(0.5))
        updated_value = get_parameter_value(vrbm, "param_1")
        @test updated_value == ComplexF64(0.5)
    end

    @testset "RBM Optimization History" begin
        using ManyVariableVariationalMonteCarlo

        # Test optimization history tracking
        vrbm = VariationalRBM{ComplexF64}(3, 4, 2)
        initialize_rbm!(vrbm.rbm, scale = 0.01)

        # Test optimization step recording
        record_optimization_step!(vrbm, 1.0, 0.1)
        @test length(vrbm.energy_history) == 1
        @test length(vrbm.gradient_history) == 1
        @test length(vrbm.parameter_history) == 1
        @test vrbm.energy_history[1] == 1.0
        @test vrbm.gradient_history[1] == 0.1

        # Test optimization statistics
        record_optimization_step!(vrbm, 0.9, 0.05)
        stats = get_optimization_statistics(vrbm)
        @test stats.n_steps == 2
        @test stats.energy_convergence > 0.0
        @test stats.gradient_convergence > 0.0

        # Test history reset
        reset_optimization_history!(vrbm)
        @test length(vrbm.energy_history) == 0
        @test length(vrbm.gradient_history) == 0
        @test length(vrbm.parameter_history) == 0
    end

    @testset "RBM Regularization" begin
        using ManyVariableVariationalMonteCarlo

        # Test RBM regularization
        vrbm = VariationalRBM{ComplexF64}(3, 4, 2)
        initialize_rbm!(vrbm.rbm, scale = 0.01)

        # Set regularization parameters
        vrbm.l1_regularization = 0.01
        vrbm.l2_regularization = 0.001

        # Test regularization application
        params_before = copy(get_rbm_parameters(vrbm.rbm))
        apply_regularization!(vrbm)
        params_after = get_rbm_parameters(vrbm.rbm)

        # Parameters should be different after regularization
        @test params_before != params_after
    end

    @testset "RBM Ensemble" begin
        using ManyVariableVariationalMonteCarlo

        # Test RBM ensemble creation
        ensemble = RBMEnsemble{ComplexF64}(3, 4, 6, 2)
        @test ensemble.n_networks == 3
        @test length(ensemble.rbms) == 3
        @test length(ensemble.weights) == 3
        @test sum(ensemble.weights) ≈ 1.0

        # Initialize all RBMs
        for rbm in ensemble.rbms
            initialize_rbm!(rbm, scale = 0.01)
        end

        # Test ensemble weight calculation
        visible_state = [1, 0, 1, 0]
        hidden_state = [1, 0, 1, 0, 1, 0]
        phys_state = [0.5, 0.3]

        ensemble_weight =
            ensemble_rbm_weight(ensemble, visible_state, hidden_state, phys_state)
        @test isa(ensemble_weight, ComplexF64)
        @test ensemble_weight != zero(ComplexF64)

        # Test ensemble weight updating
        performance_scores = [0.8, 0.9, 0.7]
        update_ensemble_weights!(ensemble, performance_scores)
        @test sum(ensemble.weights) ≈ 1.0
        @test ensemble.weights[2] > ensemble.weights[1]  # Best performance should have highest weight
        @test ensemble.weights[2] > ensemble.weights[3]
    end

    @testset "RBM Performance" begin
        using ManyVariableVariationalMonteCarlo

        # Test performance with larger networks
        rbm = RBMNetwork{ComplexF64}(10, 20, 5)
        initialize_rbm!(rbm, scale = 0.01)

        visible_state = rand([0, 1], 10)
        hidden_state = rand([0, 1], 20)
        phys_state = rand(ComplexF64, 5)

        # Benchmark weight calculations
        n_iterations = 1000
        @time begin
            for _ = 1:n_iterations
                rbm_weight_phys(rbm, visible_state, hidden_state, phys_state)
            end
        end

        # Benchmark gradient calculations
        @time begin
            for _ = 1:n_iterations
                rbm_gradient_phys(rbm, visible_state, hidden_state, phys_state)
            end
        end
    end

    @testset "RBM Edge Cases" begin
        using ManyVariableVariationalMonteCarlo

        # Test RBM with zero hidden units
        rbm = RBMNetwork{ComplexF64}(3, 0, 2)
        @test rbm.n_hidden == 0
        @test size(rbm.weights) == (0, 3)
        @test length(rbm.hidden_bias) == 0

        # Test RBM with zero physical layer
        rbm = RBMNetwork{ComplexF64}(3, 4, 0)
        @test rbm.n_phys_layer == 0
        @test size(rbm.phys_weights) == (4, 0)

        # Test parameter count for edge cases
        @test rbm_parameter_count(rbm) == 3 * 4 + 3 + 4  # weights + visible_bias + hidden_bias
    end

    @testset "RBM Complex vs Real" begin
        using ManyVariableVariationalMonteCarlo

        # Test complex RBM
        rbm_complex = RBMNetwork{ComplexF64}(3, 4, 2)
        initialize_rbm!(rbm_complex, scale = 0.01)

        visible_state = [1, 0, 1]
        hidden_state = [1, 0, 1, 0]
        phys_state = [0.5 + 0.1im, 0.3 - 0.2im]

        weight_complex =
            rbm_weight_phys(rbm_complex, visible_state, hidden_state, phys_state)
        @test isa(weight_complex, ComplexF64)
        @test imag(weight_complex) != 0.0

        # Test real RBM
        rbm_real = RBMNetwork{Float64}(3, 4, 2)
        initialize_rbm!(rbm_real, scale = 0.01)

        phys_state_real = [0.5, 0.3]
        weight_real =
            rbm_weight_phys(rbm_real, visible_state, hidden_state, phys_state_real)
        @test isa(weight_real, Float64)
        @test imag(weight_real) == 0.0
    end

    @testset "RBM Reset and Initialization" begin
        using ManyVariableVariationalMonteCarlo

        # Test RBM reset
        rbm = RBMNetwork{ComplexF64}(3, 4, 2)
        initialize_rbm!(rbm, scale = 0.01)
        @test rbm.is_initialized

        reset_rbm!(rbm)
        @test !rbm.is_initialized
        @test all(x -> x == zero(ComplexF64), rbm.weights)
        @test all(x -> x == zero(ComplexF64), rbm.visible_bias)
        @test all(x -> x == zero(ComplexF64), rbm.hidden_bias)
        @test all(x -> x == zero(ComplexF64), rbm.phys_weights)
    end
end # @testitem "rbm_enhanced"
