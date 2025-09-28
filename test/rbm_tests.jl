@testitem "rbm" begin
    @testset "RBMNetwork basic functionality" begin
        using Test
        using ManyVariableVariationalMonteCarlo
        using StableRNGs
        using Random
        # Test RBMNetwork creation
        rbm = RBMNetwork{ComplexF64}(4, 8, 2)
        @test rbm.n_visible == 4
        @test rbm.n_hidden == 8
        @test rbm.n_phys_layer == 2
        @test size(rbm.weights) == (8, 4)
        @test length(rbm.visible_bias) == 4
        @test length(rbm.hidden_bias) == 8
        @test size(rbm.phys_weights) == (8, 2)
        @test !rbm.is_initialized
        @test rbm.is_complex
    end
    @testset "RBMNetwork initialization" begin
        using Test
        using ManyVariableVariationalMonteCarlo
        using StableRNGs
        using Random
        # Test RBM initialization
        rbm = RBMNetwork{ComplexF64}(3, 6, 2)
        rng = MersenneTwister(12345)
        initialize_rbm!(rbm; rng = rng)
        @test rbm.is_initialized
        @test all(x -> abs(x) < 0.1, rbm.weights)  # Small initial weights
        @test all(x -> abs(x) < 0.1, rbm.visible_bias)
        @test all(x -> abs(x) < 0.1, rbm.hidden_bias)
        @test all(x -> abs(x) < 0.1, rbm.phys_weights)
    end
    @testset "RBMNetwork weight computation" begin
        using Test
        using ManyVariableVariationalMonteCarlo
        using StableRNGs
        using Random
        # Test RBM weight computation
        rbm = RBMNetwork{ComplexF64}(2, 3, 1)
        rng = MersenneTwister(12345)
        initialize_rbm!(rbm; rng = rng)
        visible_state = [1, 0]
        hidden_state = [1, 0, 1]
        phys_state = [0.5 + 0.3im]
        # Test basic weight computation
        weight = rbm_weight(rbm, visible_state, hidden_state)
        log_weight = log_rbm_weight(rbm, visible_state, hidden_state)
        @test isapprox(weight, exp(log_weight), rtol = 1e-10)
        @test weight isa ComplexF64
        # Test physical layer weight computation
        weight_phys = rbm_weight_phys(rbm, visible_state, hidden_state, phys_state)
        log_weight_phys = log_rbm_weight_phys(rbm, visible_state, hidden_state, phys_state)
        @test isapprox(weight_phys, exp(log_weight_phys), rtol = 1e-10)
        @test weight_phys isa ComplexF64
    end
    @testset "RBMNetwork gradient computation" begin
        using Test
        using ManyVariableVariationalMonteCarlo
        using StableRNGs
        using Random
        # Test RBM gradient computation
        rbm = RBMNetwork{ComplexF64}(2, 3, 1)
        rng = MersenneTwister(12345)
        initialize_rbm!(rbm; rng = rng)
        visible_state = [1, 0]
        hidden_state = [1, 0, 1]
        phys_state = [0.5 + 0.3im]
        # Test basic gradient computation
        grad = rbm_gradient(rbm, visible_state, hidden_state)
        @test size(grad.weights) == (3, 2)
        @test length(grad.visible_bias) == 2
        @test length(grad.hidden_bias) == 3
        @test all(x -> x isa ComplexF64, grad.weights)
        @test all(x -> x isa ComplexF64, grad.visible_bias)
        @test all(x -> x isa ComplexF64, grad.hidden_bias)
        # Test physical layer gradient computation
        grad_phys = rbm_gradient_phys(rbm, visible_state, hidden_state, phys_state)
        @test size(grad_phys.weights) == (3, 2)
        @test length(grad_phys.visible_bias) == 2
        @test length(grad_phys.hidden_bias) == 3
        @test size(grad_phys.phys_weights) == (3, 1)
        @test all(x -> x isa ComplexF64, grad_phys.phys_weights)
    end
    @testset "RBMNetwork weight updates" begin
        using Test
        using ManyVariableVariationalMonteCarlo
        using StableRNGs
        using Random
        # Test RBM weight updates
        rbm = RBMNetwork{ComplexF64}(2, 3, 1)
        rng = MersenneTwister(12345)
        initialize_rbm!(rbm; rng = rng)
        # Store original weights
        original_weights = copy(rbm.weights)
        original_visible_bias = copy(rbm.visible_bias)
        original_hidden_bias = copy(rbm.hidden_bias)
        original_phys_weights = copy(rbm.phys_weights)
        # Create dummy gradients
        grad_weights = 0.1 * ones(ComplexF64, 3, 2)
        grad_visible_bias = 0.1 * ones(ComplexF64, 2)
        grad_hidden_bias = 0.1 * ones(ComplexF64, 3)
        grad_phys_weights = 0.1 * ones(ComplexF64, 3, 1)
        # Test basic weight update
        update_rbm_weights!(
            rbm,
            grad_weights,
            grad_visible_bias,
            grad_hidden_bias;
            learning_rate = 0.01,
        )
        @test rbm.weights ≈ original_weights + 0.01 * grad_weights
        @test rbm.visible_bias ≈ original_visible_bias + 0.01 * grad_visible_bias
        @test rbm.hidden_bias ≈ original_hidden_bias + 0.01 * grad_hidden_bias
        # Test physical layer weight update
        update_rbm_weights_phys!(
            rbm,
            grad_weights,
            grad_visible_bias,
            grad_hidden_bias,
            grad_phys_weights;
            learning_rate = 0.01,
        )
        @test rbm.phys_weights ≈ original_phys_weights + 0.01 * grad_phys_weights
    end
    @testset "RBMNetwork parameter management" begin
        using Test
        using ManyVariableVariationalMonteCarlo
        using StableRNGs
        using Random
        # Test parameter management
        rbm = RBMNetwork{ComplexF64}(2, 3, 1)
        rng = MersenneTwister(12345)
        initialize_rbm!(rbm; rng = rng)
        # Test parameter count
        param_count = rbm_parameter_count(rbm)
        expected_count = 3 * 2 + 2 + 3 + 3 * 1  # weights + visible_bias + hidden_bias + phys_weights
        @test param_count == expected_count
        # Test parameter extraction
        params = get_rbm_parameters(rbm)
        @test length(params) == param_count
        @test all(x -> x isa ComplexF64, params)
        # Test parameter setting
        new_params = 0.5 * ones(ComplexF64, param_count)
        set_rbm_parameters!(rbm, new_params)
        @test all(x -> x == 0.5, rbm.weights)
        @test all(x -> x == 0.5, rbm.visible_bias)
        @test all(x -> x == 0.5, rbm.hidden_bias)
        @test all(x -> x == 0.5, rbm.phys_weights)
    end
    @testset "RBMNetwork reset" begin
        using Test
        using ManyVariableVariationalMonteCarlo
        using StableRNGs
        using Random
        # Test RBM reset
        rbm = RBMNetwork{ComplexF64}(2, 3, 1)
        rng = MersenneTwister(12345)
        initialize_rbm!(rbm; rng = rng)
        @test rbm.is_initialized
        reset_rbm!(rbm)
        @test !rbm.is_initialized
        @test all(x -> x == 0.0, rbm.weights)
        @test all(x -> x == 0.0, rbm.visible_bias)
        @test all(x -> x == 0.0, rbm.hidden_bias)
        @test all(x -> x == 0.0, rbm.phys_weights)
    end
    @testset "RBMNetwork real-valued" begin
        using Test
        using ManyVariableVariationalMonteCarlo
        using StableRNGs
        using Random
        # Test real-valued RBM
        rbm = RBMNetwork{Float64}(2, 3, 1)
        rng = MersenneTwister(12345)
        initialize_rbm!(rbm; rng = rng)
        @test rbm.is_initialized
        @test !rbm.is_complex
        @test all(x -> x isa Float64, rbm.weights)
        @test all(x -> x isa Float64, rbm.visible_bias)
        @test all(x -> x isa Float64, rbm.hidden_bias)
        @test all(x -> x isa Float64, rbm.phys_weights)
        visible_state = [1, 0]
        hidden_state = [1, 0, 1]
        phys_state = [0.5]
        weight = rbm_weight(rbm, visible_state, hidden_state)
        @test weight isa Float64
    end
end # @testitem "rbm"
