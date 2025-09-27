@testitem "optimization" begin
    """
    Tests for optimization algorithms

    Tests all optimization functionality including:
    - Stochastic reconfiguration
    - Conjugate gradient
    - Adam optimization
    - Optimization manager
    - Performance benchmarks
    """

    using Test
    using StableRNGs
    using ManyVariableVariationalMonteCarlo

    @testset "Optimization Config Basic Functionality" begin

        # Test optimization config creation
        config = OptimizationConfig()
        @test config.method == STOCHASTIC_RECONFIGURATION
        @test config.learning_rate == 0.01
        @test config.max_iterations == 1000
        @test config.convergence_tolerance == 1e-6
        @test config.regularization_parameter == 1e-4
        @test config.momentum_parameter == 0.9
        @test config.beta1 == 0.9
        @test config.beta2 == 0.999
        @test config.epsilon == 1e-8

        # Test custom config
        config_custom = OptimizationConfig(method = CONJUGATE_GRADIENT, learning_rate = 0.1)
        @test config_custom.method == CONJUGATE_GRADIENT
        @test config_custom.learning_rate == 0.1
    end

    @testset "Stochastic Reconfiguration Basic Functionality" begin

        # Test SR creation
        sr = StochasticReconfiguration{ComplexF64}(3, 5)
        @test sr.n_parameters == 3
        @test sr.n_samples == 5
        @test size(sr.overlap_matrix) == (3, 3)
        @test length(sr.force_vector) == 3
        @test size(sr.parameter_gradients) == (5, 3)
        @test length(sr.energy_values) == 5
        @test sr.total_iterations == 0
        @test isempty(sr.convergence_history)
        @test isempty(sr.energy_history)

        # Test parameter gradients and energy values
        parameter_gradients = rand(ComplexF64, 5, 3)
        energy_values = rand(ComplexF64, 5)
        weights = rand(Float64, 5)
        weights ./= sum(weights)  # Normalize weights

        # Test overlap matrix computation
        compute_overlap_matrix!(sr, parameter_gradients, weights)
        @test isa(sr.overlap_matrix, Matrix{ComplexF64})
        @test size(sr.overlap_matrix) == (3, 3)

        # Test force vector computation
        compute_force_vector!(sr, parameter_gradients, energy_values, weights)
        @test isa(sr.force_vector, Vector{ComplexF64})
        @test length(sr.force_vector) == 3
    end

    @testset "Stochastic Reconfiguration Equations" begin

        # Test SR equations solving
        sr = StochasticReconfiguration{ComplexF64}(3, 5)
        config = OptimizationConfig()

        # Set up test data
        parameter_gradients = rand(ComplexF64, 5, 3)
        energy_values = rand(ComplexF64, 5)
        weights = rand(Float64, 5)
        weights ./= sum(weights)

        # Compute matrices
        compute_overlap_matrix!(sr, parameter_gradients, weights)
        compute_force_vector!(sr, parameter_gradients, energy_values, weights)

        # Solve equations
        solve_sr_equations!(sr, config)
        @test isa(sr.parameter_delta, Vector{ComplexF64})
        @test length(sr.parameter_delta) == 3
    end

    @testset "Conjugate Gradient Basic Functionality" begin

        # Test CG creation
        cg = ConjugateGradient{ComplexF64}(3)
        @test cg.n_parameters == 3
        @test length(cg.current_parameters) == 3
        @test length(cg.current_gradient) == 3
        @test length(cg.search_direction) == 3
        @test length(cg.previous_gradient) == 3
        @test length(cg.previous_search_direction) == 3
        @test cg.total_iterations == 0
        @test isempty(cg.convergence_history)
        @test isempty(cg.gradient_norm_history)

        # Test CG step
        gradient = rand(ComplexF64, 3)
        hessian_vector = rand(ComplexF64, 3)
        config = OptimizationConfig()

        cg_step!(cg, gradient, hessian_vector, config)
        @test cg.total_iterations == 1
        @test cg.current_gradient == gradient
    end

    @testset "Adam Optimizer Basic Functionality" begin

        # Test Adam creation
        adam = AdamOptimizer{ComplexF64}(3)
        @test adam.n_parameters == 3
        @test length(adam.current_parameters) == 3
        @test length(adam.current_gradient) == 3
        @test length(adam.first_moment) == 3
        @test length(adam.second_moment) == 3
        @test adam.total_iterations == 0
        @test isempty(adam.convergence_history)

        # Test Adam step
        gradient = rand(ComplexF64, 3)
        config = OptimizationConfig()

        adam_step!(adam, gradient, config)
        @test adam.total_iterations == 1
        @test adam.current_gradient == gradient
    end

    @testset "Optimization Manager Basic Functionality" begin

        # Test optimization manager creation
        config = OptimizationConfig()
        manager = OptimizationManager{ComplexF64}(3, 5, config)
        @test manager.sr.n_parameters == 3
        @test manager.sr.n_samples == 5
        @test manager.cg.n_parameters == 3
        @test manager.adam.n_parameters == 3
        @test manager.config == config
        @test length(manager.current_parameters) == 3
        @test length(manager.current_gradient) == 3
        @test manager.current_energy == zero(ComplexF64)
        @test manager.total_optimization_steps == 0
        @test manager.optimization_time == 0.0
        @test !manager.convergence_achieved
    end

    @testset "Optimization Manager Parameter Optimization" begin

        # Test parameter optimization
        config = OptimizationConfig(method = STOCHASTIC_RECONFIGURATION)
        manager = OptimizationManager{ComplexF64}(3, 5, config)

        # Set up test data
        parameter_gradients = rand(ComplexF64, 5, 3)
        energy_values = rand(ComplexF64, 5)
        weights = rand(Float64, 5)
        weights ./= sum(weights)

        # Optimize parameters
        optimize_parameters!(manager, parameter_gradients, energy_values, weights)
        @test manager.total_optimization_steps == 1
        @test isa(manager.current_parameters, Vector{ComplexF64})
        @test length(manager.current_parameters) == 3
    end

    @testset "Optimization Manager Statistics" begin

        # Test optimization statistics
        config = OptimizationConfig()
        manager = OptimizationManager{ComplexF64}(3, 5, config)

        # Add some optimization steps
        parameter_gradients = rand(ComplexF64, 5, 3)
        energy_values = rand(ComplexF64, 5)
        weights = rand(Float64, 5)
        weights ./= sum(weights)

        for _ = 1:5
            optimize_parameters!(manager, parameter_gradients, energy_values, weights)
        end

        # Test statistics retrieval
        stats = get_optimization_statistics(manager)
        @test stats.total_steps == 5
        @test isa(stats.current_energy, ComplexF64)
        @test isa(stats.gradient_norm, Float64)
        @test isa(stats.optimization_time, Float64)
    end

    @testset "Optimization Manager Reset" begin

        # Test optimization reset
        config = OptimizationConfig()
        manager = OptimizationManager{ComplexF64}(3, 5, config)

        # Add some optimization steps
        parameter_gradients = rand(ComplexF64, 5, 3)
        energy_values = rand(ComplexF64, 5)
        weights = rand(Float64, 5)
        weights ./= sum(weights)

        optimize_parameters!(manager, parameter_gradients, energy_values, weights)

        # Reset optimization
        reset_optimization!(manager)
        @test manager.sr.total_iterations == 0
        @test manager.cg.total_iterations == 0
        @test manager.adam.total_iterations == 0
        @test manager.total_optimization_steps == 0
        @test manager.optimization_time == 0.0
        @test !manager.convergence_achieved
    end

    @testset "Optimization Performance" begin

        # Test performance with larger systems
        config = OptimizationConfig()
        manager = OptimizationManager{ComplexF64}(10, 100, config)

        parameter_gradients = rand(ComplexF64, 100, 10)
        energy_values = rand(ComplexF64, 100)
        weights = rand(Float64, 100)
        weights ./= sum(weights)

        # Benchmark optimization
        n_iterations = 100
        @time begin
            for _ = 1:n_iterations
                optimize_parameters!(manager, parameter_gradients, energy_values, weights)
            end
        end

        @test manager.total_optimization_steps == n_iterations
    end

    @testset "Optimization Edge Cases" begin

        # Test edge cases
        config = OptimizationConfig()
        manager = OptimizationManager{ComplexF64}(0, 0, config)  # Zero parameters

        parameter_gradients = zeros(ComplexF64, 0, 0)
        energy_values = ComplexF64[]
        weights = Float64[]

        # Should not crash
        optimize_parameters!(manager, parameter_gradients, energy_values, weights)
        @test manager.total_optimization_steps == 1
    end

    @testset "Optimization Complex vs Real" begin

        # Test complex optimization
        config_complex = OptimizationConfig()
        manager_complex = OptimizationManager{ComplexF64}(3, 5, config_complex)

        parameter_gradients_complex = rand(ComplexF64, 5, 3)
        energy_values_complex = rand(ComplexF64, 5)
        weights = rand(Float64, 5)
        weights ./= sum(weights)

        optimize_parameters!(
            manager_complex,
            parameter_gradients_complex,
            energy_values_complex,
            weights,
        )
        @test isa(manager_complex.current_parameters, Vector{ComplexF64})

        # Test real optimization
        config_real = OptimizationConfig()
        manager_real = OptimizationManager{Float64}(3, 5, config_real)

        parameter_gradients_real = rand(Float64, 5, 3)
        energy_values_real = rand(Float64, 5)

        optimize_parameters!(
            manager_real,
            parameter_gradients_real,
            energy_values_real,
            weights,
        )
        @test isa(manager_real.current_parameters, Vector{Float64})
    end

    @testset "Optimization Methods Comparison" begin

        # Test different optimization methods
        methods = [STOCHASTIC_RECONFIGURATION, CONJUGATE_GRADIENT, ADAM]

        for method in methods
            config = OptimizationConfig(method = method)
            manager = OptimizationManager{ComplexF64}(3, 5, config)

            parameter_gradients = rand(ComplexF64, 5, 3)
            energy_values = rand(ComplexF64, 5)
            weights = rand(Float64, 5)
            weights ./= sum(weights)

            # Optimize parameters
            optimize_parameters!(manager, parameter_gradients, energy_values, weights)
            @test manager.total_optimization_steps == 1
            @test isa(manager.current_parameters, Vector{ComplexF64})
        end
    end

    @testset "Optimization Convergence" begin

        # Test convergence detection
        config = OptimizationConfig(convergence_tolerance = 1e-3)
        manager = OptimizationManager{ComplexF64}(3, 5, config)

        # Set up test data with small gradient
        parameter_gradients = rand(ComplexF64, 5, 3) * 1e-4  # Small gradients
        energy_values = rand(ComplexF64, 5)
        weights = rand(Float64, 5)
        weights ./= sum(weights)

        # Optimize parameters
        optimize_parameters!(manager, parameter_gradients, energy_values, weights)

        # Check if convergence is detected
        stats = get_optimization_statistics(manager)
        @test stats.gradient_norm < 1e-3
    end
end # @testitem "optimization"
