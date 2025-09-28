"""
Stochastic Reconfiguration Optimization for ManyVariableVariationalMonteCarlo.jl

Implements optimization methods including:
- Stochastic reconfiguration (SR) method
- Conjugate gradient solver
- Parameter gradient calculations
- Diagonal preconditioning
- Optimization history tracking

Ported from stcopt*.c in the C reference implementation.
"""

using LinearAlgebra
using SparseArrays
using Random
using StableRNGs

"""
    OptimizationMethod

Enumeration of different optimization methods.
"""
@enum OptimizationMethod begin
    STOCHASTIC_RECONFIGURATION
    CONJUGATE_GRADIENT
    ADAM
    RMSPROP
    MOMENTUM
end

"""
    OptimizationConfig

Configuration for optimization algorithms.
"""
mutable struct OptimizationConfig
    method::OptimizationMethod
    learning_rate::Float64
    max_iterations::Int
    convergence_tolerance::Float64
    regularization_parameter::Float64
    momentum_parameter::Float64
    beta1::Float64  # For Adam
    beta2::Float64  # For Adam
    epsilon::Float64  # For Adam/RMSprop
    # SR-CG options
    use_sr_cg::Bool
    sr_cg_max_iter::Int
    sr_cg_tol::Float64

    function OptimizationConfig(;
        method::OptimizationMethod = STOCHASTIC_RECONFIGURATION,
        learning_rate::Float64 = 0.01,
        max_iterations::Int = 1000,
        convergence_tolerance::Float64 = 1e-6,
        regularization_parameter::Float64 = 1e-4,
        momentum_parameter::Float64 = 0.9,
        beta1::Float64 = 0.9,
        beta2::Float64 = 0.999,
        epsilon::Float64 = 1e-8,
        use_sr_cg::Bool = false,
        sr_cg_max_iter::Int = 0,
        sr_cg_tol::Float64 = 1e-6,
    )
        new(
            method,
            learning_rate,
            max_iterations,
            convergence_tolerance,
            regularization_parameter,
            momentum_parameter,
            beta1,
            beta2,
            epsilon,
            use_sr_cg,
            sr_cg_max_iter,
            sr_cg_tol,
        )
    end
end

"""
    StochasticReconfiguration{T}

Stochastic reconfiguration optimization method.
"""
mutable struct StochasticReconfiguration{T<:Union{Float64,ComplexF64}}
    # System parameters
    n_parameters::Int
    n_samples::Int

    # SR matrices
    overlap_matrix::Matrix{T}  # S matrix
    force_vector::Vector{T}    # F vector
    parameter_gradients::Matrix{T}  # Gradients for each sample
    energy_values::Vector{T}   # Energy values for each sample

    # Working arrays
    work_matrix::Matrix{T}
    work_vector::Vector{T}
    parameter_delta::Vector{T}

    # Statistics
    total_iterations::Int
    convergence_history::Vector{Float64}
    energy_history::Vector{Float64}

    function StochasticReconfiguration{T}(n_parameters::Int, n_samples::Int) where {T}
        overlap_matrix = zeros(T, n_parameters, n_parameters)
        force_vector = zeros(T, n_parameters)
        parameter_gradients = zeros(T, n_samples, n_parameters)
        energy_values = zeros(T, n_samples)

        work_matrix = zeros(T, n_parameters, n_parameters)
        work_vector = zeros(T, n_parameters)
        parameter_delta = zeros(T, n_parameters)

        new{T}(
            n_parameters,
            n_samples,
            overlap_matrix,
            force_vector,
            parameter_gradients,
            energy_values,
            work_matrix,
            work_vector,
            parameter_delta,
            0,
            Float64[],
            Float64[],
        )
    end
end

"""
    compute_overlap_matrix!(sr::StochasticReconfiguration{T},
                           parameter_gradients::Matrix{T},
                           weights::Vector{Float64}) where T

Compute the overlap matrix S for stochastic reconfiguration.
"""
function compute_overlap_matrix!(
    sr::StochasticReconfiguration{T},
    parameter_gradients::Matrix{T},
    weights::Vector{Float64},
) where {T}
    n_samples = size(parameter_gradients, 1)
    n_params = size(parameter_gradients, 2)

    # Initialize overlap matrix
    fill!(sr.overlap_matrix, zero(T))

    # Compute S_ij = <O_i^* O_j> - <O_i^*><O_j>
    for i = 1:n_params
        for j = 1:n_params
            # Compute <O_i^* O_j>
            sum_oi_oj = zero(T)
            sum_oi = zero(T)
            sum_oj = zero(T)
            sum_weights = 0.0

            for k = 1:n_samples
                weight = weights[k]
                oi = parameter_gradients[k, i]
                oj = parameter_gradients[k, j]

                sum_oi_oj += weight * conj(oi) * oj
                sum_oi += weight * conj(oi)
                sum_oj += weight * oj
                sum_weights += weight
            end

            if sum_weights > 0
                sr.overlap_matrix[i, j] =
                    (sum_oi_oj / sum_weights) -
                    (sum_oi / sum_weights) * (sum_oj / sum_weights)
            end
        end
    end
end

"""
    compute_force_vector!(sr::StochasticReconfiguration{T},
                         parameter_gradients::Matrix{T},
                         energy_values::Vector{T},
                         weights::Vector{Float64}) where T

Compute the force vector F for stochastic reconfiguration.
"""
function compute_force_vector!(
    sr::StochasticReconfiguration{T},
    parameter_gradients::Matrix{T},
    energy_values::Vector{T},
    weights::Vector{Float64},
) where {T}
    n_samples = length(energy_values)
    n_params = size(parameter_gradients, 2)

    # Initialize force vector
    fill!(sr.force_vector, zero(T))

    # Compute F_i = <O_i^* E> - <O_i^*><E>
    for i = 1:n_params
        sum_oi_e = zero(T)
        sum_oi = zero(T)
        sum_e = zero(T)
        sum_weights = 0.0

        for k = 1:n_samples
            weight = weights[k]
            oi = parameter_gradients[k, i]
            e = energy_values[k]

            sum_oi_e += weight * conj(oi) * e
            sum_oi += weight * conj(oi)
            sum_e += weight * e
            sum_weights += weight
        end

        if sum_weights > 0
            sr.force_vector[i] =
                (sum_oi_e / sum_weights) - (sum_oi / sum_weights) * (sum_e / sum_weights)
        end
    end
end

"""
    solve_sr_equations!(sr::StochasticReconfiguration{T},
                       config::OptimizationConfig) where T

Solve the stochastic reconfiguration equations S * δp = F.
"""
function solve_sr_equations!(
    sr::StochasticReconfiguration{T},
    config::OptimizationConfig,
) where {T}
    n_params = sr.n_parameters

    # Add regularization to diagonal elements
    for i = 1:n_params
        sr.work_matrix[i, i] = sr.overlap_matrix[i, i] + config.regularization_parameter
    end

    # Copy off-diagonal elements
    for i = 1:n_params
        for j = 1:n_params
            if i != j
                sr.work_matrix[i, j] = sr.overlap_matrix[i, j]
            end
        end
    end

    # Solve linear system
    try
        # Use LU decomposition for stability
        lu_result = lu!(sr.work_matrix)
        sr.parameter_delta .= lu_result \ sr.force_vector
    catch e
        # Fallback to least squares if matrix is singular
        println("Warning: SR matrix is singular, using least squares solution")
        sr.parameter_delta .= sr.work_matrix \ sr.force_vector
    end
end

"""
    solve_sr_equations_cg!(sr::StochasticReconfiguration{T}, config::OptimizationConfig) where T

Solve the SR equations S * δp = F using a simple conjugate gradient on the
regularized system (S + λI) δp = F with diagonal preconditioning.
"""
function solve_sr_equations_cg!(
    sr::StochasticReconfiguration{T},
    config::OptimizationConfig,
) where {T}
    n = sr.n_parameters
    # Regularized matrix-vector product closure: y = (S+λI) x
    λ = config.regularization_parameter
    function matvec!(y::AbstractVector{T}, x::AbstractVector{T})
        @inbounds begin
            for i in 1:n
                acc = λ * x[i]
                @simd for j in 1:n
                    acc += sr.overlap_matrix[i, j] * x[j]
                end
                y[i] = acc
            end
        end
        return y
    end

    # Diagonal preconditioner M ≈ diag(S+λI)
    Mdiag = similar(sr.work_vector)
    @inbounds for i in 1:n
        Mdiag[i] = sr.overlap_matrix[i, i] + λ
        if Mdiag[i] == 0
            Mdiag[i] = one(T)
        end
    end
    function precond!(z::AbstractVector{T}, r::AbstractVector{T})
        @inbounds for i in 1:n
            z[i] = r[i] / Mdiag[i]
        end
        return z
    end

    x = sr.parameter_delta
    r = sr.work_vector; fill!(r, zero(T))
    Ap = similar(r)
    z = similar(r)
    p = similar(r)

    # r0 = F - A x0 (x0 initially zeros)
    copyto!(r, sr.force_vector)
    precond!(z, r)
    copyto!(p, z)
    rz_old = dot(conj.(r), z)

    itmax = max(1, config.sr_cg_max_iter)
    tol2 = config.sr_cg_tol^2
    for it in 1:itmax
        matvec!(Ap, p)
        α = rz_old / dot(conj.(p), Ap)
        @inbounds @simd for i in 1:n
            x[i] += α * p[i]
            r[i] -= α * Ap[i]
        end
        # Check convergence ||r||^2
        if real(dot(conj.(r), r)) <= tol2
            break
        end
        precond!(z, r)
        rz_new = dot(conj.(r), z)
        β = rz_new / rz_old
        @inbounds @simd for i in 1:n
            p[i] = z[i] + β * p[i]
        end
        rz_old = rz_new
    end
end

"""
    ConjugateGradient{T}

Conjugate gradient optimization method.
"""
mutable struct ConjugateGradient{T<:Union{Float64,ComplexF64}}
    # System parameters
    n_parameters::Int

    # CG state
    current_parameters::Vector{T}
    current_gradient::Vector{T}
    search_direction::Vector{T}
    previous_gradient::Vector{T}
    previous_search_direction::Vector{T}

    # Working arrays
    work_vector::Vector{T}
    hessian_vector::Vector{T}

    # Statistics
    total_iterations::Int
    convergence_history::Vector{Float64}
    gradient_norm_history::Vector{Float64}

    function ConjugateGradient{T}(n_parameters::Int) where {T}
        current_parameters = zeros(T, n_parameters)
        current_gradient = zeros(T, n_parameters)
        search_direction = zeros(T, n_parameters)
        previous_gradient = zeros(T, n_parameters)
        previous_search_direction = zeros(T, n_parameters)

        work_vector = zeros(T, n_parameters)
        hessian_vector = zeros(T, n_parameters)

        new{T}(
            n_parameters,
            current_parameters,
            current_gradient,
            search_direction,
            previous_gradient,
            previous_search_direction,
            work_vector,
            hessian_vector,
            0,
            Float64[],
            Float64[],
        )
    end
end

"""
    cg_step!(cg::ConjugateGradient{T}, gradient::Vector{T},
             hessian_vector::Vector{T}, config::OptimizationConfig) where T

Perform one conjugate gradient step.
"""
function cg_step!(
    cg::ConjugateGradient{T},
    gradient::Vector{T},
    hessian_vector::Vector{T},
    config::OptimizationConfig,
) where {T}
    n_params = cg.n_parameters

    # Compute beta (Polak-Ribiere formula)
    if cg.total_iterations > 0
        beta = max(
            0.0,
            real(
                dot(gradient, gradient - cg.previous_gradient) /
                dot(cg.previous_gradient, cg.previous_gradient),
            ),
        )
    else
        beta = 0.0
    end

    # Update search direction
    cg.search_direction .= gradient + beta * cg.previous_search_direction

    # Compute step size using line search (simplified)
    step_size = config.learning_rate

    # Update parameters
    cg.current_parameters .+= step_size * cg.search_direction

    # Store previous values
    cg.previous_gradient .= cg.current_gradient
    cg.previous_search_direction .= cg.search_direction
    cg.current_gradient .= gradient

    cg.total_iterations += 1
end

"""
    AdamOptimizer{T}

Adam optimization method.
"""
mutable struct AdamOptimizer{T<:Union{Float64,ComplexF64}}
    # System parameters
    n_parameters::Int

    # Adam state
    current_parameters::Vector{T}
    current_gradient::Vector{T}
    first_moment::Vector{T}
    second_moment::Vector{T}

    # Working arrays
    work_vector::Vector{T}

    # Statistics
    total_iterations::Int
    convergence_history::Vector{Float64}

    function AdamOptimizer{T}(n_parameters::Int) where {T}
        current_parameters = zeros(T, n_parameters)
        current_gradient = zeros(T, n_parameters)
        first_moment = zeros(T, n_parameters)
        second_moment = zeros(T, n_parameters)

        work_vector = zeros(T, n_parameters)

        new{T}(
            n_parameters,
            current_parameters,
            current_gradient,
            first_moment,
            second_moment,
            work_vector,
            0,
            Float64[],
        )
    end
end

"""
    adam_step!(adam::AdamOptimizer{T}, gradient::Vector{T},
               config::OptimizationConfig) where T

Perform one Adam optimization step.
"""
function adam_step!(
    adam::AdamOptimizer{T},
    gradient::Vector{T},
    config::OptimizationConfig,
) where {T}
    n_params = adam.n_parameters

    # Update biased first moment estimate
    adam.first_moment .= config.beta1 * adam.first_moment + (1 - config.beta1) * gradient

    # Update biased second moment estimate
    adam.second_moment .=
        config.beta2 * adam.second_moment + (1 - config.beta2) * (gradient .* gradient)

    # Compute bias-corrected first moment estimate
    first_moment_corrected = adam.first_moment / (1 - config.beta1^adam.total_iterations)

    # Compute bias-corrected second moment estimate
    second_moment_corrected = adam.second_moment / (1 - config.beta2^adam.total_iterations)

    # Update parameters
    for i = 1:n_params
        # Ensure second moment is positive and not too small
        second_moment_val = max(real(second_moment_corrected[i]), config.epsilon^2)
        adam.current_parameters[i] -=
            config.learning_rate * first_moment_corrected[i] /
            (sqrt(second_moment_val) + config.epsilon)
    end

    adam.current_gradient .= gradient
    adam.total_iterations += 1
end

"""
    OptimizationManager{T}

Manages optimization algorithms and parameter updates.
"""
mutable struct OptimizationManager{T<:Union{Float64,ComplexF64}}
    # Optimization methods
    sr::StochasticReconfiguration{T}
    cg::ConjugateGradient{T}
    adam::AdamOptimizer{T}

    # Configuration
    config::OptimizationConfig

    # Current state
    current_parameters::Vector{T}
    current_gradient::Vector{T}
    current_energy::T

    # Statistics
    total_optimization_steps::Int
    optimization_time::Float64
    convergence_achieved::Bool

    function OptimizationManager{T}(
        n_parameters::Int,
        n_samples::Int,
        config::OptimizationConfig,
    ) where {T}
        sr = StochasticReconfiguration{T}(n_parameters, n_samples)
        cg = ConjugateGradient{T}(n_parameters)
        adam = AdamOptimizer{T}(n_parameters)

        current_parameters = zeros(T, n_parameters)
        current_gradient = zeros(T, n_parameters)
        current_energy = zero(T)

        new{T}(
            sr,
            cg,
            adam,
            config,
            current_parameters,
            current_gradient,
            current_energy,
            0,
            0.0,
            false,
        )
    end
end

"""
    optimize_parameters!(manager::OptimizationManager{T},
                        parameter_gradients::Matrix{T},
                        energy_values::Vector{T},
                        weights::Vector{Float64}) where T

Optimize parameters using the configured method.
"""
function optimize_parameters!(
    manager::OptimizationManager{T},
    parameter_gradients::Matrix{T},
    energy_values::Vector{T},
    weights::Vector{Float64},
) where {T}
    n_params = length(manager.current_parameters)
    n_samples = length(energy_values)

    if manager.config.method == STOCHASTIC_RECONFIGURATION
        # Stochastic reconfiguration
        compute_overlap_matrix!(manager.sr, parameter_gradients, weights)
        compute_force_vector!(manager.sr, parameter_gradients, energy_values, weights)
        solve_sr_equations!(manager.sr, manager.config)

        # Update parameters
        manager.current_parameters .+= manager.sr.parameter_delta

    elseif manager.config.method == CONJUGATE_GRADIENT
        # Conjugate gradient
        # Compute average gradient
        fill!(manager.current_gradient, zero(T))
        for i = 1:n_samples
            weight = weights[i]
            manager.current_gradient .+= weight * parameter_gradients[i, :]
        end
        manager.current_gradient ./= sum(weights)

        # Perform CG step
        cg_step!(manager.cg, manager.current_gradient, zeros(T, n_params), manager.config)
        manager.current_parameters .= manager.cg.current_parameters

    elseif manager.config.method == ADAM
        # Adam optimization
        # Compute average gradient
        fill!(manager.current_gradient, zero(T))
        for i = 1:n_samples
            weight = weights[i]
            manager.current_gradient .+= weight * parameter_gradients[i, :]
        end
        manager.current_gradient ./= sum(weights)

        # Perform Adam step
        adam_step!(manager.adam, manager.current_gradient, manager.config)
        manager.current_parameters .= manager.adam.current_parameters
    end

    # Update statistics
    manager.total_optimization_steps += 1

    # Check convergence
    if manager.total_optimization_steps > 1
        gradient_norm = sqrt(real(dot(manager.current_gradient, manager.current_gradient)))
        if gradient_norm < manager.config.convergence_tolerance
            manager.convergence_achieved = true
        end
    end
end

"""
    get_optimization_statistics(manager::OptimizationManager{T}) where T

Get optimization statistics.
"""
function get_optimization_statistics(manager::OptimizationManager{T}) where {T}
    return (
        total_steps = manager.total_optimization_steps,
        convergence_achieved = manager.convergence_achieved,
        current_energy = manager.current_energy,
        gradient_norm = sqrt(real(dot(manager.current_gradient, manager.current_gradient))),
        optimization_time = manager.optimization_time,
    )
end

"""
    reset_optimization!(manager::OptimizationManager{T}) where T

Reset optimization state.
"""
function reset_optimization!(manager::OptimizationManager{T}) where {T}
    # Reset SR
    fill!(manager.sr.overlap_matrix, zero(T))
    fill!(manager.sr.force_vector, zero(T))
    fill!(manager.sr.parameter_delta, zero(T))
    manager.sr.total_iterations = 0
    empty!(manager.sr.convergence_history)
    empty!(manager.sr.energy_history)

    # Reset CG
    fill!(manager.cg.current_parameters, zero(T))
    fill!(manager.cg.current_gradient, zero(T))
    fill!(manager.cg.search_direction, zero(T))
    fill!(manager.cg.previous_gradient, zero(T))
    fill!(manager.cg.previous_search_direction, zero(T))
    manager.cg.total_iterations = 0
    empty!(manager.cg.convergence_history)
    empty!(manager.cg.gradient_norm_history)

    # Reset Adam
    fill!(manager.adam.current_parameters, zero(T))
    fill!(manager.adam.current_gradient, zero(T))
    fill!(manager.adam.first_moment, zero(T))
    fill!(manager.adam.second_moment, zero(T))
    manager.adam.total_iterations = 0
    empty!(manager.adam.convergence_history)

    # Reset manager state
    fill!(manager.current_parameters, zero(T))
    fill!(manager.current_gradient, zero(T))
    manager.current_energy = zero(T)
    manager.total_optimization_steps = 0
    manager.optimization_time = 0.0
    manager.convergence_achieved = false
end

"""
    benchmark_optimization(n_parameters::Int = 100, n_samples::Int = 1000, n_iterations::Int = 100)

Benchmark optimization algorithms.
"""
function benchmark_optimization(
    n_parameters::Int = 100,
    n_samples::Int = 1000,
    n_iterations::Int = 100,
)
    println(
        "Benchmarking optimization algorithms (n_params=$n_parameters, n_samples=$n_samples, iterations=$n_iterations)...",
    )

    # Test different optimization methods
    methods = [STOCHASTIC_RECONFIGURATION, CONJUGATE_GRADIENT, ADAM]

    for method in methods
        println("  Testing $(method)...")

        config = OptimizationConfig(
            method = method,
            learning_rate = 0.01,
            max_iterations = n_iterations,
        )
        manager = OptimizationManager{ComplexF64}(n_parameters, n_samples, config)

        # Generate random test data
        parameter_gradients = rand(ComplexF64, n_samples, n_parameters)
        energy_values = rand(ComplexF64, n_samples)
        weights = rand(Float64, n_samples)
        weights ./= sum(weights)  # Normalize weights

        # Benchmark optimization
        @time begin
            for _ = 1:n_iterations
                optimize_parameters!(manager, parameter_gradients, energy_values, weights)
            end
        end

        stats = get_optimization_statistics(manager)
        println("    Total steps: $(stats.total_steps)")
        println("    Convergence achieved: $(stats.convergence_achieved)")
        println("    Gradient norm: $(stats.gradient_norm)")
    end

    println("Optimization benchmark completed.")
end
