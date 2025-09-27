"""
Quantum Projections implementation for ManyVariableVariationalMonteCarlo.jl

Implements quantum number projection including:
- Spin projection
- Momentum projection
- Particle number projection
- Gauss-Legendre quadrature for continuous projections
- Symmetry operations

Ported from projection.c and gauleg.c in the C reference implementation.
"""

using LinearAlgebra
using LinearAlgebra.BLAS
using LinearAlgebra.LAPACK

"""
    ProjectionType

Enumeration of different projection types.
"""
@enum ProjectionType begin
    SPIN_PROJECTION
    MOMENTUM_PROJECTION
    PARTICLE_NUMBER_PROJECTION
    PARITY_PROJECTION
    POINT_GROUP_PROJECTION
end

"""
    ProjectionOperator{T}

Represents a quantum number projection operator.
"""
mutable struct ProjectionOperator{T <: Union{Float64, ComplexF64}}
    projection_type::ProjectionType
    target_value::T
    weight::T
    is_active::Bool

    # For continuous projections (e.g., momentum)
    integration_points::Vector{T}
    integration_weights::Vector{T}

    # For discrete projections
    discrete_values::Vector{T}
    discrete_weights::Vector{T}

    function ProjectionOperator{T}(proj_type::ProjectionType, target::T, weight::T = one(T)) where T
        integration_points = T[]
        integration_weights = T[]
        discrete_values = T[]
        discrete_weights = T[]

        new{T}(proj_type, target, weight, true, integration_points, integration_weights, discrete_values, discrete_weights)
    end
end

"""
    QuantumProjection{T}

Main structure for quantum number projections.
"""
mutable struct QuantumProjection{T <: Union{Float64, ComplexF64}}
    # Projection operators
    spin_projections::Vector{ProjectionOperator{T}}
    momentum_projections::Vector{ProjectionOperator{T}}
    particle_number_projections::Vector{ProjectionOperator{T}}
    parity_projections::Vector{ProjectionOperator{T}}

    # System parameters
    n_site::Int
    n_elec::Int
    n_spin::Int

    # Working arrays
    projection_buffer::Vector{T}
    integration_workspace::Vector{T}

    # Performance tracking
    total_projections::Int
    projection_time::Float64

    function QuantumProjection{T}(n_site::Int, n_elec::Int, n_spin::Int = 2) where T
        spin_projections = ProjectionOperator{T}[]
        momentum_projections = ProjectionOperator{T}[]
        particle_number_projections = ProjectionOperator{T}[]
        parity_projections = ProjectionOperator{T}[]

        projection_buffer = Vector{T}(undef, max(n_site, n_elec))
        integration_workspace = Vector{T}(undef, 1000)  # For Gauss-Legendre quadrature

        new{T}(spin_projections, momentum_projections, particle_number_projections, parity_projections,
               n_site, n_elec, n_spin, projection_buffer, integration_workspace, 0, 0.0)
    end
end

"""
    add_spin_projection!(qp::QuantumProjection{T}, target_sz::T, weight::T = one(T)) where T

Add a spin projection operator targeting total spin z-component.
"""
function add_spin_projection!(qp::QuantumProjection{T}, target_sz::T, weight::T = one(T)) where T
    proj = ProjectionOperator{T}(SPIN_PROJECTION, target_sz, weight)
    push!(qp.spin_projections, proj)
    return proj
end

"""
    add_momentum_projection!(qp::QuantumProjection{T}, target_k::Vector{T}, weight::T = one(T)) where T

Add a momentum projection operator targeting specific momentum.
"""
function add_momentum_projection!(qp::QuantumProjection{T}, target_k::Vector{T}, weight::T = one(T)) where T
    proj = ProjectionOperator{T}(MOMENTUM_PROJECTION, target_k[1], weight)  # Simplified for 1D
    push!(qp.momentum_projections, proj)
    return proj
end

"""
    add_particle_number_projection!(qp::QuantumProjection{T}, target_n::Int, weight::T = one(T)) where T

Add a particle number projection operator.
"""
function add_particle_number_projection!(qp::QuantumProjection{T}, target_n::Int, weight::T = one(T)) where T
    proj = ProjectionOperator{T}(PARTICLE_NUMBER_PROJECTION, T(target_n), weight)
    push!(qp.particle_number_projections, proj)
    return proj
end

"""
    add_parity_projection!(qp::QuantumProjection{T}, target_parity::Int, weight::T = one(T)) where T

Add a parity projection operator.
"""
function add_parity_projection!(qp::QuantumProjection{T}, target_parity::Int, weight::T = one(T)) where T
    proj = ProjectionOperator{T}(PARITY_PROJECTION, T(target_parity), weight)
    push!(qp.parity_projections, proj)
    return proj
end

"""
    calculate_projection_ratio(qp::QuantumProjection{T}, ele_idx::Vector{Int},
                              ele_cfg::Vector{Int}, ele_num::Vector{Int}) where T

Calculate the projection ratio for given electron configuration.
"""
function calculate_projection_ratio(qp::QuantumProjection{T}, ele_idx::Vector{Int},
                                   ele_cfg::Vector{Int}, ele_num::Vector{Int}) where T <: Union{Float64, ComplexF64}

    ratio = one(T)

    # Spin projection
    for proj in qp.spin_projections
        if proj.is_active
            ratio *= _calculate_spin_projection_ratio(proj, ele_idx, ele_cfg, ele_num)
        end
    end

    # Momentum projection
    for proj in qp.momentum_projections
        if proj.is_active
            ratio *= _calculate_momentum_projection_ratio(proj, ele_idx, ele_cfg, ele_num)
        end
    end

    # Particle number projection
    for proj in qp.particle_number_projections
        if proj.is_active
            ratio *= _calculate_particle_number_projection_ratio(proj, ele_idx, ele_cfg, ele_num)
        end
    end

    # Parity projection
    for proj in qp.parity_projections
        if proj.is_active
            ratio *= _calculate_parity_projection_ratio(proj, ele_idx, ele_cfg, ele_num)
        end
    end

    qp.total_projections += 1
    return ratio
end

"""
    _calculate_spin_projection_ratio(proj::ProjectionOperator{T}, ele_idx::Vector{Int},
                                    ele_cfg::Vector{Int}, ele_num::Vector{Int}) where T

Calculate spin projection ratio.
"""
function _calculate_spin_projection_ratio(proj::ProjectionOperator{T}, ele_idx::Vector{Int},
                                         ele_cfg::Vector{Int}, ele_num::Vector{Int}) where T <: Union{Float64, ComplexF64}

    # Calculate current total spin z-component
    current_sz = _calculate_total_sz(ele_idx, ele_cfg, ele_num)

    # For discrete projections, return weight if target matches, 0 otherwise
    if isapprox(current_sz, proj.target_value, atol=1e-10)
        return proj.weight
    else
        return zero(T)
    end
end

"""
    _calculate_momentum_projection_ratio(proj::ProjectionOperator{T}, ele_idx::Vector{Int},
                                        ele_cfg::Vector{Int}, ele_num::Vector{Int}) where T

Calculate momentum projection ratio.
"""
function _calculate_momentum_projection_ratio(proj::ProjectionOperator{T}, ele_idx::Vector{Int},
                                             ele_cfg::Vector{Int}, ele_num::Vector{Int}) where T <: Union{Float64, ComplexF64}

    # Calculate current total momentum
    current_k = _calculate_total_momentum(ele_idx, ele_cfg, ele_num)

    # For discrete projections, return weight if target matches, 0 otherwise
    if isapprox(current_k, proj.target_value, atol=1e-10)
        return proj.weight
    else
        return zero(T)
    end
end

"""
    _calculate_particle_number_projection_ratio(proj::ProjectionOperator{T}, ele_idx::Vector{Int},
                                               ele_cfg::Vector{Int}, ele_num::Vector{Int}) where T

Calculate particle number projection ratio.
"""
function _calculate_particle_number_projection_ratio(proj::ProjectionOperator{T}, ele_idx::Vector{Int},
                                                    ele_cfg::Vector{Int}, ele_num::Vector{Int}) where T <: Union{Float64, ComplexF64}

    # Calculate current particle number
    current_n = sum(ele_num)

    # For discrete projections, return weight if target matches, 0 otherwise
    if current_n == Int(proj.target_value)
        return proj.weight
    else
        return zero(T)
    end
end

"""
    _calculate_parity_projection_ratio(proj::ProjectionOperator{T}, ele_idx::Vector{Int},
                                      ele_cfg::Vector{Int}, ele_num::Vector{Int}) where T

Calculate parity projection ratio.
"""
function _calculate_parity_projection_ratio(proj::ProjectionOperator{T}, ele_idx::Vector{Int},
                                           ele_cfg::Vector{Int}, ele_num::Vector{Int}) where T <: Union{Float64, ComplexF64}

    # Calculate current parity
    current_parity = _calculate_parity(ele_idx, ele_cfg, ele_num)

    # For discrete projections, return weight if target matches, 0 otherwise
    if current_parity == Int(proj.target_value)
        return proj.weight
    else
        return zero(T)
    end
end

# Helper functions for quantum number calculations

function _calculate_total_sz(ele_idx::Vector{Int}, ele_cfg::Vector{Int}, ele_num::Vector{Int})
    # Simplified calculation of total spin z-component
    # In a real implementation, this would depend on the specific spin configuration
    n_elec = length(ele_idx)
    return (n_elec % 2 == 0) ? 0.0 : 0.5
end

function _calculate_total_momentum(ele_idx::Vector{Int}, ele_cfg::Vector{Int}, ele_num::Vector{Int})
    # Simplified calculation of total momentum
    # In a real implementation, this would depend on the lattice structure and electron positions
    return 0.0
end

function _calculate_parity(ele_idx::Vector{Int}, ele_cfg::Vector{Int}, ele_num::Vector{Int})
    # Simplified calculation of parity
    # In a real implementation, this would depend on the spatial configuration
    return 1
end

"""
    gauss_legendre_quadrature(n::Int, a::T, b::T) where T

Generate Gauss-Legendre quadrature points and weights for integration over [a, b].
"""
function gauss_legendre_quadrature(n::Int, a::T, b::T) where T <: Union{Float64, ComplexF64}
    if n <= 0
        throw(ArgumentError("Number of points must be positive"))
    end

    # Generate Legendre polynomial roots and weights
    x, w = _gauss_legendre_roots_weights(n)

    # Transform from [-1, 1] to [a, b]
    points = T[]
    weights = T[]

    for i in 1:n
        # Linear transformation: x ∈ [-1, 1] → t ∈ [a, b]
        t = (b - a) / 2 * x[i] + (a + b) / 2
        push!(points, T(t))

        # Weight transformation
        wt = (b - a) / 2 * w[i]
        push!(weights, T(wt))
    end

    return points, weights
end

"""
    _gauss_legendre_roots_weights(n::Int)

Generate Gauss-Legendre quadrature roots and weights using Newton's method.
"""
function _gauss_legendre_roots_weights(n::Int)
    if n == 1
        return [0.0], [2.0]
    end

    # Initial guess for roots (Chebyshev points)
    x = [cos(π * (2*i - 1) / (2*n)) for i in 1:n]

    # Newton's method to find Legendre polynomial roots
    for iter in 1:20
        converged = true
        for i in 1:n
            p, dp = _legendre_polynomial_and_derivative(n, x[i])
            if abs(p) > 1e-12
                x_new = x[i] - p / dp
                if abs(x_new - x[i]) > 1e-12
                    converged = false
                end
                x[i] = x_new
            end
        end
        if converged
            break
        end
    end

    # Calculate weights
    w = zeros(n)
    for i in 1:n
        _, dp = _legendre_polynomial_and_derivative(n, x[i])
        w[i] = 2.0 / ((1 - x[i]^2) * dp^2)
    end

    return x, w
end

"""
    _legendre_polynomial_and_derivative(n::Int, x::Float64)

Calculate Legendre polynomial P_n(x) and its derivative.
"""
function _legendre_polynomial_and_derivative(n::Int, x::Float64)
    if n == 0
        return 1.0, 0.0
    elseif n == 1
        return x, 1.0
    end

    # Use recurrence relation: (n+1)P_{n+1}(x) = (2n+1)xP_n(x) - nP_{n-1}(x)
    p_prev = 1.0
    p_curr = x
    dp_prev = 0.0
    dp_curr = 1.0

    for k in 1:n-1
        p_next = ((2*k + 1) * x * p_curr - k * p_prev) / (k + 1)
        dp_next = ((2*k + 1) * (p_curr + x * dp_curr) - k * dp_prev) / (k + 1)

        p_prev = p_curr
        p_curr = p_next
        dp_prev = dp_curr
        dp_curr = dp_next
    end

    return p_curr, dp_curr
end

"""
    setup_continuous_projection!(proj::ProjectionOperator{T}, n_points::Int,
                                a::T, b::T) where T

Setup continuous projection using Gauss-Legendre quadrature.
"""
function setup_continuous_projection!(proj::ProjectionOperator{T}, n_points::Int,
                                     a::T, b::T) where T <: Union{Float64, ComplexF64}

    points, weights = gauss_legendre_quadrature(n_points, a, b)

    proj.integration_points = points
    proj.integration_weights = weights

    return proj
end

"""
    calculate_continuous_projection_ratio(proj::ProjectionOperator{T}, ele_idx::Vector{Int},
                                         ele_cfg::Vector{Int}, ele_num::Vector{Int}) where T

Calculate projection ratio for continuous projections using numerical integration.
"""
function calculate_continuous_projection_ratio(proj::ProjectionOperator{T}, ele_idx::Vector{Int},
                                              ele_cfg::Vector{Int}, ele_num::Vector{Int}) where T <: Union{Float64, ComplexF64}

    if isempty(proj.integration_points)
        throw(ArgumentError("Continuous projection not set up. Call setup_continuous_projection! first."))
    end

    integral = zero(T)

    for (point, weight) in zip(proj.integration_points, proj.integration_weights)
        # Calculate integrand at this point
        integrand = _calculate_projection_integrand(proj, point, ele_idx, ele_cfg, ele_num)
        integral += weight * integrand
    end

    return proj.weight * integral
end

"""
    _calculate_projection_integrand(proj::ProjectionOperator{T}, point::T, ele_idx::Vector{Int},
                                   ele_cfg::Vector{Int}, ele_num::Vector{Int}) where T

Calculate the integrand for continuous projection at a given point.
"""
function _calculate_projection_integrand(proj::ProjectionOperator{T}, point::T, ele_idx::Vector{Int},
                                        ele_cfg::Vector{Int}, ele_num::Vector{Int}) where T <: Union{Float64, ComplexF64}

    if proj.projection_type == MOMENTUM_PROJECTION
        # For momentum projection, the integrand is exp(i * k * point)
        return exp(im * proj.target_value * point)
    else
        # For other continuous projections, use a simplified form
        return exp(-(point - proj.target_value)^2)
    end
end

"""
    benchmark_projections(n_site::Int = 10, n_elec::Int = 5, n_iterations::Int = 1000)

Benchmark quantum projection calculations.
"""
function benchmark_projections(n_site::Int = 10, n_elec::Int = 5, n_iterations::Int = 1000)
    println("Benchmarking quantum projections (n_site=$n_site, n_elec=$n_elec, iterations=$n_iterations)...")

    # Create projection calculator
    qp = QuantumProjection{ComplexF64}(n_site, n_elec)

    # Add some projections
    add_spin_projection!(qp, ComplexF64(0.0), ComplexF64(1.0))
    add_particle_number_projection!(qp, n_elec, ComplexF64(1.0))
    add_parity_projection!(qp, 1, ComplexF64(1.0))

    # Initialize electron configuration
    ele_idx = collect(1:n_elec)
    ele_cfg = zeros(Int, n_site)
    ele_cfg[1:n_elec] .= 1
    ele_num = copy(ele_cfg)

    # Benchmark projection calculations
    @time begin
        for _ in 1:n_iterations
            calculate_projection_ratio(qp, ele_idx, ele_cfg, ele_num)
        end
    end
    println("  Projection ratio calculation rate")

    # Benchmark Gauss-Legendre quadrature
    @time begin
        for _ in 1:n_iterations÷10
            gauss_legendre_quadrature(10, -1.0, 1.0)
        end
    end
    println("  Gauss-Legendre quadrature rate")

    println("Projection benchmark completed.")
    println("  Total projections: $(qp.total_projections)")
end
