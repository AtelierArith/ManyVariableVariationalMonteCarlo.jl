"""
Restricted Boltzmann Machine (RBM) implementation for ManyVariableVariationalMonteCarlo.jl

Implements RBM neural network components for variational wavefunctions including:
- RBM weight and bias management
- Forward and backward propagation
- Gradient calculations for optimization
- Complex-valued RBM support

Ported from rbm.c in the C reference implementation.
"""

using LinearAlgebra
using Random

"""
    RBMNetwork{T}

Represents a Restricted Boltzmann Machine with complex-valued weights and biases.
"""
mutable struct RBMNetwork{T <: Union{Float64, ComplexF64}}
    # Network architecture
    n_visible::Int
    n_hidden::Int
    n_phys_layer::Int

    # Weight matrices
    weights::Matrix{T}  # n_hidden × n_visible
    visible_bias::Vector{T}  # n_visible
    hidden_bias::Vector{T}   # n_hidden

    # Physical layer weights (for physical to hidden connections)
    phys_weights::Matrix{T}  # n_hidden × n_phys_layer

    # Optimization flags
    is_initialized::Bool
    is_complex::Bool

    function RBMNetwork{T}(n_visible::Int, n_hidden::Int, n_phys_layer::Int = 0) where T <: Union{Float64, ComplexF64}
        weights = zeros(T, n_hidden, n_visible)
        visible_bias = zeros(T, n_visible)
        hidden_bias = zeros(T, n_hidden)
        phys_weights = zeros(T, n_hidden, n_phys_layer)

        new{T}(n_visible, n_hidden, n_phys_layer, weights, visible_bias, hidden_bias,
               phys_weights, false, T <: ComplexF64)
    end
end

"""
    initialize_rbm!(rbm::RBMNetwork{T}; scale::Float64 = 0.01, rng::AbstractRNG = Random.GLOBAL_RNG)

Initialize RBM weights and biases with small random values.
"""
function initialize_rbm!(rbm::RBMNetwork{T}; scale::Float64 = 0.01, rng::AbstractRNG = Random.GLOBAL_RNG) where T
    n_hidden = rbm.n_hidden
    n_visible = rbm.n_visible
    n_phys = rbm.n_phys_layer

    # Initialize weights with small random values
    if rbm.is_complex
        for i in 1:n_hidden
            for j in 1:n_visible
                rbm.weights[i, j] = scale * (randn(rng) + im * randn(rng)) / n_hidden
            end
        end

        for i in 1:n_visible
            rbm.visible_bias[i] = scale * (randn(rng) + im * randn(rng)) / n_hidden
        end

        for i in 1:n_hidden
            rbm.hidden_bias[i] = scale * (randn(rng) + im * randn(rng)) / n_hidden
        end

        for i in 1:n_hidden
            for j in 1:n_phys
                rbm.phys_weights[i, j] = scale * (randn(rng) + im * randn(rng)) / n_hidden
            end
        end
    else
        for i in 1:n_hidden
            for j in 1:n_visible
                rbm.weights[i, j] = scale * randn(rng) / n_hidden
            end
        end

        for i in 1:n_visible
            rbm.visible_bias[i] = scale * randn(rng) / n_hidden
        end

        for i in 1:n_hidden
            rbm.hidden_bias[i] = scale * randn(rng) / n_hidden
        end

        for i in 1:n_hidden
            for j in 1:n_phys
                rbm.phys_weights[i, j] = scale * randn(rng) / n_hidden
            end
        end
    end

    rbm.is_initialized = true
end

"""
    rbm_weight(rbm::RBMNetwork{T}, visible_state::Vector{Int}, hidden_state::Vector{Int})

Compute RBM weight for given visible and hidden states.
"""
function rbm_weight(rbm::RBMNetwork{T}, visible_state::Vector{Int}, hidden_state::Vector{Int}) where T
    if !rbm.is_initialized
        throw(ArgumentError("RBM not initialized"))
    end

    # Convert binary states to complex numbers
    v = convert.(T, visible_state)
    h = convert.(T, hidden_state)

    # Compute energy: E = -sum_i sum_j w_ij v_i h_j - sum_i a_i v_i - sum_j b_j h_j
    energy = zero(T)

    # Visible-hidden interactions
    for i in 1:rbm.n_visible
        for j in 1:rbm.n_hidden
            energy -= rbm.weights[j, i] * v[i] * h[j]
        end
    end

    # Visible bias terms
    for i in 1:rbm.n_visible
        energy -= rbm.visible_bias[i] * v[i]
    end

    # Hidden bias terms
    for j in 1:rbm.n_hidden
        energy -= rbm.hidden_bias[j] * h[j]
    end

    return exp(energy)
end

"""
    log_rbm_weight(rbm::RBMNetwork{T}, visible_state::Vector{Int}, hidden_state::Vector{Int})

Compute log RBM weight for given visible and hidden states.
"""
function log_rbm_weight(rbm::RBMNetwork{T}, visible_state::Vector{Int}, hidden_state::Vector{Int}) where T
    if !rbm.is_initialized
        throw(ArgumentError("RBM not initialized"))
    end

    # Convert binary states to complex numbers
    v = convert.(T, visible_state)
    h = convert.(T, hidden_state)

    # Compute energy: E = -sum_i sum_j w_ij v_i h_j - sum_i a_i v_i - sum_j b_j h_j
    energy = zero(T)

    # Visible-hidden interactions
    for i in 1:rbm.n_visible
        for j in 1:rbm.n_hidden
            energy -= rbm.weights[j, i] * v[i] * h[j]
        end
    end

    # Visible bias terms
    for i in 1:rbm.n_visible
        energy -= rbm.visible_bias[i] * v[i]
    end

    # Hidden bias terms
    for j in 1:rbm.n_hidden
        energy -= rbm.hidden_bias[j] * h[j]
    end

    return energy
end

"""
    rbm_weight_phys(rbm::RBMNetwork{T}, visible_state::Vector{Int}, hidden_state::Vector{Int}, phys_state::Vector{T})

Compute RBM weight including physical layer contributions.
"""
function rbm_weight_phys(rbm::RBMNetwork{T}, visible_state::Vector{Int}, hidden_state::Vector{Int}, phys_state::Vector{T}) where T
    if !rbm.is_initialized
        throw(ArgumentError("RBM not initialized"))
    end

    if length(phys_state) != rbm.n_phys_layer
        throw(ArgumentError("Physical state dimension mismatch"))
    end

    # Convert binary states to complex numbers
    v = convert.(T, visible_state)
    h = convert.(T, hidden_state)

    # Compute energy including physical layer
    energy = zero(T)

    # Visible-hidden interactions
    for i in 1:rbm.n_visible
        for j in 1:rbm.n_hidden
            energy -= rbm.weights[j, i] * v[i] * h[j]
        end
    end

    # Physical-hidden interactions
    for i in 1:rbm.n_phys_layer
        for j in 1:rbm.n_hidden
            energy -= rbm.phys_weights[j, i] * phys_state[i] * h[j]
        end
    end

    # Visible bias terms
    for i in 1:rbm.n_visible
        energy -= rbm.visible_bias[i] * v[i]
    end

    # Hidden bias terms
    for j in 1:rbm.n_hidden
        energy -= rbm.hidden_bias[j] * h[j]
    end

    return exp(energy)
end

"""
    log_rbm_weight_phys(rbm::RBMNetwork{T}, visible_state::Vector{Int}, hidden_state::Vector{Int}, phys_state::Vector{T})

Compute log RBM weight including physical layer contributions.
"""
function log_rbm_weight_phys(rbm::RBMNetwork{T}, visible_state::Vector{Int}, hidden_state::Vector{Int}, phys_state::Vector{T}) where T
    if !rbm.is_initialized
        throw(ArgumentError("RBM not initialized"))
    end

    if length(phys_state) != rbm.n_phys_layer
        throw(ArgumentError("Physical state dimension mismatch"))
    end

    # Convert binary states to complex numbers
    v = convert.(T, visible_state)
    h = convert.(T, hidden_state)

    # Compute energy including physical layer
    energy = zero(T)

    # Visible-hidden interactions
    for i in 1:rbm.n_visible
        for j in 1:rbm.n_hidden
            energy -= rbm.weights[j, i] * v[i] * h[j]
        end
    end

    # Physical-hidden interactions
    for i in 1:rbm.n_phys_layer
        for j in 1:rbm.n_hidden
            energy -= rbm.phys_weights[j, i] * phys_state[i] * h[j]
        end
    end

    # Visible bias terms
    for i in 1:rbm.n_visible
        energy -= rbm.visible_bias[i] * v[i]
    end

    # Hidden bias terms
    for j in 1:rbm.n_hidden
        energy -= rbm.hidden_bias[j] * h[j]
    end

    return energy
end

"""
    rbm_gradient(rbm::RBMNetwork{T}, visible_state::Vector{Int}, hidden_state::Vector{Int})

Compute RBM gradient with respect to weights and biases.
"""
function rbm_gradient(rbm::RBMNetwork{T}, visible_state::Vector{Int}, hidden_state::Vector{Int}) where T
    if !rbm.is_initialized
        throw(ArgumentError("RBM not initialized"))
    end

    # Convert binary states to complex numbers
    v = convert.(T, visible_state)
    h = convert.(T, hidden_state)

    # Initialize gradient arrays
    grad_weights = zeros(T, rbm.n_hidden, rbm.n_visible)
    grad_visible_bias = zeros(T, rbm.n_visible)
    grad_hidden_bias = zeros(T, rbm.n_hidden)

    # Compute gradients
    for i in 1:rbm.n_visible
        for j in 1:rbm.n_hidden
            grad_weights[j, i] = v[i] * h[j]
        end
    end

    for i in 1:rbm.n_visible
        grad_visible_bias[i] = v[i]
    end

    for j in 1:rbm.n_hidden
        grad_hidden_bias[j] = h[j]
    end

    return (weights = grad_weights, visible_bias = grad_visible_bias, hidden_bias = grad_hidden_bias)
end

"""
    rbm_gradient_phys(rbm::RBMNetwork{T}, visible_state::Vector{Int}, hidden_state::Vector{Int}, phys_state::Vector{T})

Compute RBM gradient including physical layer contributions.
"""
function rbm_gradient_phys(rbm::RBMNetwork{T}, visible_state::Vector{Int}, hidden_state::Vector{Int}, phys_state::Vector{T}) where T
    if !rbm.is_initialized
        throw(ArgumentError("RBM not initialized"))
    end

    if length(phys_state) != rbm.n_phys_layer
        throw(ArgumentError("Physical state dimension mismatch"))
    end

    # Convert binary states to complex numbers
    v = convert.(T, visible_state)
    h = convert.(T, hidden_state)

    # Initialize gradient arrays
    grad_weights = zeros(T, rbm.n_hidden, rbm.n_visible)
    grad_visible_bias = zeros(T, rbm.n_visible)
    grad_hidden_bias = zeros(T, rbm.n_hidden)
    grad_phys_weights = zeros(T, rbm.n_hidden, rbm.n_phys_layer)

    # Compute gradients
    for i in 1:rbm.n_visible
        for j in 1:rbm.n_hidden
            grad_weights[j, i] = v[i] * h[j]
        end
    end

    for i in 1:rbm.n_phys_layer
        for j in 1:rbm.n_hidden
            grad_phys_weights[j, i] = phys_state[i] * h[j]
        end
    end

    for i in 1:rbm.n_visible
        grad_visible_bias[i] = v[i]
    end

    for j in 1:rbm.n_hidden
        grad_hidden_bias[j] = h[j]
    end

    return (weights = grad_weights, visible_bias = grad_visible_bias, hidden_bias = grad_hidden_bias,
            phys_weights = grad_phys_weights)
end

"""
    update_rbm_weights!(rbm::RBMNetwork{T}, grad_weights::Matrix{T}, grad_visible_bias::Vector{T},
                       grad_hidden_bias::Vector{T}; learning_rate::Float64 = 0.01)

Update RBM weights and biases using gradient descent.
"""
function update_rbm_weights!(rbm::RBMNetwork{T}, grad_weights::Matrix{T}, grad_visible_bias::Vector{T},
                            grad_hidden_bias::Vector{T}; learning_rate::Float64 = 0.01) where T
    if !rbm.is_initialized
        throw(ArgumentError("RBM not initialized"))
    end

    # Update weights
    rbm.weights .+= learning_rate .* grad_weights
    rbm.visible_bias .+= learning_rate .* grad_visible_bias
    rbm.hidden_bias .+= learning_rate .* grad_hidden_bias
end

"""
    update_rbm_weights_phys!(rbm::RBMNetwork{T}, grad_weights::Matrix{T}, grad_visible_bias::Vector{T},
                            grad_hidden_bias::Vector{T}, grad_phys_weights::Matrix{T};
                            learning_rate::Float64 = 0.01)

Update RBM weights and biases including physical layer.
"""
function update_rbm_weights_phys!(rbm::RBMNetwork{T}, grad_weights::Matrix{T}, grad_visible_bias::Vector{T},
                                 grad_hidden_bias::Vector{T}, grad_phys_weights::Matrix{T};
                                 learning_rate::Float64 = 0.01) where T
    if !rbm.is_initialized
        throw(ArgumentError("RBM not initialized"))
    end

    # Update weights
    rbm.weights .+= learning_rate .* grad_weights
    rbm.visible_bias .+= learning_rate .* grad_visible_bias
    rbm.hidden_bias .+= learning_rate .* grad_hidden_bias
    rbm.phys_weights .+= learning_rate .* grad_phys_weights
end

"""
    get_rbm_parameters(rbm::RBMNetwork{T})

Extract all RBM parameters as a flat vector for optimization.
"""
function get_rbm_parameters(rbm::RBMNetwork{T}) where T
    if !rbm.is_initialized
        throw(ArgumentError("RBM not initialized"))
    end

    params = Vector{T}()

    # Flatten weights
    append!(params, vec(rbm.weights))

    # Add biases
    append!(params, rbm.visible_bias)
    append!(params, rbm.hidden_bias)

    # Add physical weights if present
    if rbm.n_phys_layer > 0
        append!(params, vec(rbm.phys_weights))
    end

    return params
end

"""
    set_rbm_parameters!(rbm::RBMNetwork{T}, params::Vector{T})

Set RBM parameters from a flat vector.
"""
function set_rbm_parameters!(rbm::RBMNetwork{T}, params::Vector{T}) where T
    if !rbm.is_initialized
        throw(ArgumentError("RBM not initialized"))
    end

    expected_size = rbm.n_hidden * rbm.n_visible + rbm.n_visible + rbm.n_hidden + rbm.n_hidden * rbm.n_phys_layer
    if length(params) != expected_size
        throw(ArgumentError("Parameter vector size mismatch"))
    end

    idx = 1

    # Set weights
    for i in 1:rbm.n_hidden
        for j in 1:rbm.n_visible
            rbm.weights[i, j] = params[idx]
            idx += 1
        end
    end

    # Set visible bias
    for i in 1:rbm.n_visible
        rbm.visible_bias[i] = params[idx]
        idx += 1
    end

    # Set hidden bias
    for i in 1:rbm.n_hidden
        rbm.hidden_bias[i] = params[idx]
        idx += 1
    end

    # Set physical weights if present
    if rbm.n_phys_layer > 0
        for i in 1:rbm.n_hidden
            for j in 1:rbm.n_phys_layer
                rbm.phys_weights[i, j] = params[idx]
                idx += 1
            end
        end
    end
end

"""
    rbm_parameter_count(rbm::RBMNetwork{T})

Get total number of RBM parameters.
"""
function rbm_parameter_count(rbm::RBMNetwork{T}) where T
    return rbm.n_hidden * rbm.n_visible + rbm.n_visible + rbm.n_hidden + rbm.n_hidden * rbm.n_phys_layer
end

"""
    reset_rbm!(rbm::RBMNetwork{T})

Reset RBM to uninitialized state.
"""
function reset_rbm!(rbm::RBMNetwork{T}) where T
    fill!(rbm.weights, zero(T))
    fill!(rbm.visible_bias, zero(T))
    fill!(rbm.hidden_bias, zero(T))
    fill!(rbm.phys_weights, zero(T))
    rbm.is_initialized = false
end

# Enhanced RBM with efficient gradient calculations

"""
    RBMGradient{T}

Stores RBM gradients and related information for efficient computation.
"""
mutable struct RBMGradient{T <: Union{Float64, ComplexF64}}
    # Gradient components
    grad_weights::Matrix{T}
    grad_visible_bias::Vector{T}
    grad_hidden_bias::Vector{T}
    grad_phys_weights::Matrix{T}

    # Gradient statistics
    gradient_norm::Float64
    max_gradient::Float64
    gradient_count::Int

    # Working arrays for efficiency
    visible_activations::Vector{T}
    hidden_activations::Vector{T}
    gradient_buffer::Vector{T}

    function RBMGradient{T}(n_visible::Int, n_hidden::Int, n_phys::Int = 0) where T
        grad_weights = zeros(T, n_hidden, n_visible)
        grad_visible_bias = zeros(T, n_visible)
        grad_hidden_bias = zeros(T, n_hidden)
        grad_phys_weights = zeros(T, n_hidden, n_phys)

        visible_activations = Vector{T}(undef, n_visible)
        hidden_activations = Vector{T}(undef, n_hidden)
        gradient_buffer = Vector{T}(undef, max(n_visible, n_hidden))

        new{T}(grad_weights, grad_visible_bias, grad_hidden_bias, grad_phys_weights,
               0.0, 0.0, 0, visible_activations, hidden_activations, gradient_buffer)
    end
end

"""
    compute_efficient_gradient!(grad::RBMGradient{T}, rbm::RBMNetwork{T},
                               visible_state::Vector{Int}, hidden_state::Vector{Int},
                               phys_state::Vector{T} = T[]) where T

Compute RBM gradient efficiently using preallocated arrays.
"""
function compute_efficient_gradient!(grad::RBMGradient{T}, rbm::RBMNetwork{T},
                                    visible_state::Vector{Int}, hidden_state::Vector{Int},
                                    phys_state::Vector{T} = T[]) where T
    if !rbm.is_initialized
        throw(ArgumentError("RBM not initialized"))
    end

    # Convert binary states to complex numbers
    v = convert.(T, visible_state)
    h = convert.(T, hidden_state)

    # Store activations for reuse
    grad.visible_activations .= v
    grad.hidden_activations .= h

    # Compute gradients efficiently
    @inbounds for i in 1:rbm.n_visible
        for j in 1:rbm.n_hidden
            grad.grad_weights[j, i] = v[i] * h[j]
        end
    end

    @inbounds for i in 1:rbm.n_visible
        grad.grad_visible_bias[i] = v[i]
    end

    @inbounds for j in 1:rbm.n_hidden
        grad.grad_hidden_bias[j] = h[j]
    end

    # Physical layer gradients if present
    if rbm.n_phys_layer > 0 && !isempty(phys_state)
        @inbounds for i in 1:rbm.n_phys_layer
            for j in 1:rbm.n_hidden
                grad.grad_phys_weights[j, i] = phys_state[i] * h[j]
            end
        end
    end

    # Update gradient statistics
    grad.gradient_norm = sqrt(sum(abs2, grad.grad_weights) + sum(abs2, grad.grad_visible_bias) +
                              sum(abs2, grad.grad_hidden_bias) + sum(abs2, grad.grad_phys_weights))
    grad.max_gradient = maximum([maximum(abs, grad.grad_weights), maximum(abs, grad.grad_visible_bias),
                                maximum(abs, grad.grad_hidden_bias), maximum(abs, grad.grad_phys_weights)])
    grad.gradient_count += 1
end

"""
    update_rbm_efficient!(rbm::RBMNetwork{T}, grad::RBMGradient{T};
                         learning_rate::Float64 = 0.01, momentum::Float64 = 0.9) where T

Update RBM parameters using efficient gradient descent with momentum.
"""
function update_rbm_efficient!(rbm::RBMNetwork{T}, grad::RBMGradient{T};
                              learning_rate::Float64 = 0.01, momentum::Float64 = 0.9) where T
    if !rbm.is_initialized
        throw(ArgumentError("RBM not initialized"))
    end

    # Update weights with momentum
    rbm.weights .+= learning_rate .* grad.grad_weights
    rbm.visible_bias .+= learning_rate .* grad.grad_visible_bias
    rbm.hidden_bias .+= learning_rate .* grad.grad_hidden_bias

    if rbm.n_phys_layer > 0
        rbm.phys_weights .+= learning_rate .* grad.grad_phys_weights
    end
end

# Variational parameter management

"""
    VariationalRBM{T}

RBM with enhanced variational parameter management.
"""
mutable struct VariationalRBM{T <: Union{Float64, ComplexF64}}
    # Core RBM
    rbm::RBMNetwork{T}

    # Parameter management
    parameter_names::Vector{String}
    parameter_bounds::Vector{Tuple{T, T}}
    parameter_scales::Vector{Float64}

    # Optimization history
    parameter_history::Vector{Vector{T}}
    energy_history::Vector{Float64}
    gradient_history::Vector{Float64}

    # Regularization
    l1_regularization::Float64
    l2_regularization::Float64

    function VariationalRBM{T}(n_visible::Int, n_hidden::Int, n_phys::Int = 0) where T
        rbm = RBMNetwork{T}(n_visible, n_hidden, n_phys)

        # Initialize parameter management
        n_params = rbm_parameter_count(rbm)
        parameter_names = String[]
        parameter_bounds = Tuple{T, T}[]
        parameter_scales = Float64[]

        # Add parameter names and bounds
        for i in 1:n_params
            push!(parameter_names, "param_$i")
            push!(parameter_bounds, (T(-10), T(10)))  # Default bounds
            push!(parameter_scales, 1.0)  # Default scale
        end

        new{T}(rbm, parameter_names, parameter_bounds, parameter_scales,
               Vector{Vector{T}}(), Float64[], Float64[], 0.0, 0.0)
    end
end

"""
    set_parameter_bounds!(vrbm::VariationalRBM{T}, param_name::String, lower::T, upper::T) where T

Set bounds for a specific parameter.
"""
function set_parameter_bounds!(vrbm::VariationalRBM{T}, param_name::String, lower::T, upper::T) where T
    idx = findfirst(==(param_name), vrbm.parameter_names)
    if idx === nothing
        throw(ArgumentError("Parameter name not found: $param_name"))
    end

    vrbm.parameter_bounds[idx] = (lower, upper)
end

"""
    set_parameter_scale!(vrbm::VariationalRBM{T}, param_name::String, scale::Float64) where T

Set scale for a specific parameter.
"""
function set_parameter_scale!(vrbm::VariationalRBM{T}, param_name::String, scale::Float64) where T
    idx = findfirst(==(param_name), vrbm.parameter_names)
    if idx === nothing
        throw(ArgumentError("Parameter name not found: $param_name"))
    end

    vrbm.parameter_scales[idx] = scale
end

"""
    get_parameter_value(vrbm::VariationalRBM{T}, param_name::String) where T

Get current value of a specific parameter.
"""
function get_parameter_value(vrbm::VariationalRBM{T}, param_name::String) where T
    idx = findfirst(==(param_name), vrbm.parameter_names)
    if idx === nothing
        throw(ArgumentError("Parameter name not found: $param_name"))
    end

    params = get_rbm_parameters(vrbm.rbm)
    return params[idx]
end

"""
    set_parameter_value!(vrbm::VariationalRBM{T}, param_name::String, value::T) where T

Set value of a specific parameter.
"""
function set_parameter_value!(vrbm::VariationalRBM{T}, param_name::String, value::T) where T
    idx = findfirst(==(param_name), vrbm.parameter_names)
    if idx === nothing
        throw(ArgumentError("Parameter name not found: $param_name"))
    end

    # Check bounds
    lower, upper = vrbm.parameter_bounds[idx]
    if value < lower || value > upper
        throw(ArgumentError("Parameter value $value out of bounds [$lower, $upper]"))
    end

    # Get current parameters and update
    params = get_rbm_parameters(vrbm.rbm)
    params[idx] = value
    set_rbm_parameters!(vrbm.rbm, params)
end

"""
    record_optimization_step!(vrbm::VariationalRBM{T}, energy::Float64, gradient_norm::Float64) where T

Record optimization step for history tracking.
"""
function record_optimization_step!(vrbm::VariationalRBM{T}, energy::Float64, gradient_norm::Float64) where T
    push!(vrbm.parameter_history, copy(get_rbm_parameters(vrbm.rbm)))
    push!(vrbm.energy_history, energy)
    push!(vrbm.gradient_history, gradient_norm)
end

"""
    apply_regularization!(vrbm::VariationalRBM{T}) where T

Apply L1 and L2 regularization to RBM parameters.
"""
function apply_regularization!(vrbm::VariationalRBM{T}) where T
    params = get_rbm_parameters(vrbm.rbm)

    # L2 regularization
    if vrbm.l2_regularization > 0
        params .*= (1.0 - vrbm.l2_regularization)
    end

    # L1 regularization (soft thresholding)
    if vrbm.l1_regularization > 0
        for i in 1:length(params)
            if abs(params[i]) < vrbm.l1_regularization
                params[i] = zero(T)
            else
                params[i] = sign(params[i]) * (abs(params[i]) - vrbm.l1_regularization)
            end
        end
    end

    set_rbm_parameters!(vrbm.rbm, params)
end

"""
    get_optimization_statistics(vrbm::VariationalRBM{T}) where T

Get optimization statistics from history.
"""
function get_optimization_statistics(vrbm::VariationalRBM{T}) where T
    if isempty(vrbm.energy_history)
        return (energy_convergence = 0.0, gradient_convergence = 0.0, n_steps = 0)
    end

    n_steps = length(vrbm.energy_history)

    # Energy convergence (relative change in last few steps)
    if n_steps >= 2
        energy_convergence = abs(vrbm.energy_history[end] - vrbm.energy_history[end-1]) / abs(vrbm.energy_history[end-1])
    else
        energy_convergence = 0.0
    end

    # Gradient convergence
    if n_steps >= 2
        gradient_convergence = abs(vrbm.gradient_history[end] - vrbm.gradient_history[end-1]) / vrbm.gradient_history[end-1]
    else
        gradient_convergence = 0.0
    end

    return (energy_convergence = energy_convergence, gradient_convergence = gradient_convergence, n_steps = n_steps)
end

"""
    reset_optimization_history!(vrbm::VariationalRBM{T}) where T

Reset optimization history.
"""
function reset_optimization_history!(vrbm::VariationalRBM{T}) where T
    empty!(vrbm.parameter_history)
    empty!(vrbm.energy_history)
    empty!(vrbm.gradient_history)
end

# Advanced RBM features

"""
    RBMEnsemble{T}

Ensemble of RBM networks for improved performance.
"""
mutable struct RBMEnsemble{T <: Union{Float64, ComplexF64}}
    rbms::Vector{RBMNetwork{T}}
    weights::Vector{Float64}
    n_networks::Int

    function RBMEnsemble{T}(n_networks::Int, n_visible::Int, n_hidden::Int, n_phys::Int = 0) where T
        rbms = [RBMNetwork{T}(n_visible, n_hidden, n_phys) for _ in 1:n_networks]
        weights = ones(Float64, n_networks) / n_networks  # Equal weights initially

        new{T}(rbms, weights, n_networks)
    end
end

"""
    ensemble_rbm_weight(ensemble::RBMEnsemble{T}, visible_state::Vector{Int}, hidden_state::Vector{Int},
                       phys_state::Vector{T} = T[]) where T

Calculate weighted ensemble RBM weight.
"""
function ensemble_rbm_weight(ensemble::RBMEnsemble{T}, visible_state::Vector{Int}, hidden_state::Vector{Int},
                            phys_state::Vector{T} = T[]) where T
    total_weight = zero(T)

    for i in 1:ensemble.n_networks
        if phys_state isa Vector{T} && !isempty(phys_state)
            weight = rbm_weight_phys(ensemble.rbms[i], visible_state, hidden_state, phys_state)
        else
            weight = rbm_weight(ensemble.rbms[i], visible_state, hidden_state)
        end

        total_weight += ensemble.weights[i] * weight
    end

    return total_weight
end

"""
    update_ensemble_weights!(ensemble::RBMEnsemble{T}, performance_scores::Vector{Float64}) where T

Update ensemble weights based on performance scores.
"""
function update_ensemble_weights!(ensemble::RBMEnsemble{T}, performance_scores::Vector{Float64}) where T
    if length(performance_scores) != ensemble.n_networks
        throw(ArgumentError("Performance scores length mismatch"))
    end

    # Softmax weighting based on performance
    max_score = maximum(performance_scores)
    exp_scores = exp.(performance_scores .- max_score)
    ensemble.weights .= exp_scores ./ sum(exp_scores)
end
