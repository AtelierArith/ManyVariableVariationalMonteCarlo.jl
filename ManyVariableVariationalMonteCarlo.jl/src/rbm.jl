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
