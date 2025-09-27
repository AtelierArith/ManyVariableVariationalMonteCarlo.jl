"""
Jastrow Factor implementation for ManyVariableVariationalMonteCarlo.jl

Implements Jastrow correlation factors including:
- Two-body Jastrow factors
- Three-body Jastrow factors
- Optimized update algorithms
- Various correlation forms (Gutzwiller, density-density, etc.)
- Complex-valued Jastrow factors

Ported from jastrow.c and related files in the C reference implementation.
"""

using LinearAlgebra
using Random

"""
    JastrowType

Enumeration of different Jastrow factor types.
"""
@enum JastrowType begin
    GUTZWILLER
    DENSITY_DENSITY
    SPIN_SPIN
    THREE_BODY
    MOMENTUM_DEPENDENT
    CUSTOM
end

"""
    JastrowParameter{T}

Represents a single Jastrow parameter with metadata.
"""
mutable struct JastrowParameter{T<:Union{Float64,ComplexF64}}
    value::T
    is_active::Bool
    parameter_type::JastrowType
    site_indices::Vector{Int}
    spin_indices::Vector{Int}

    function JastrowParameter{T}(
        value::T,
        param_type::JastrowType,
        site_indices::Vector{Int} = Int[],
        spin_indices::Vector{Int} = Int[],
    ) where {T}
        new{T}(value, true, param_type, site_indices, spin_indices)
    end
end

"""
    JastrowFactor{T}

Main structure for Jastrow correlation factors.
"""
mutable struct JastrowFactor{T<:Union{Float64,ComplexF64}}
    # Parameters
    gutzwiller_params::Vector{JastrowParameter{T}}
    density_density_params::Vector{JastrowParameter{T}}
    spin_spin_params::Vector{JastrowParameter{T}}
    three_body_params::Vector{JastrowParameter{T}}

    # System parameters
    n_site::Int
    n_elec::Int
    n_spin::Int

    # Working arrays for efficiency
    correlation_buffer::Vector{T}
    update_buffer::Vector{T}
    gradient_buffer::Vector{T}

    # Performance tracking
    total_evaluations::Int
    evaluation_time::Float64

    function JastrowFactor{T}(n_site::Int, n_elec::Int, n_spin::Int = 2) where {T}
        gutzwiller_params = JastrowParameter{T}[]
        density_density_params = JastrowParameter{T}[]
        spin_spin_params = JastrowParameter{T}[]
        three_body_params = JastrowParameter{T}[]

        correlation_buffer = Vector{T}(undef, n_site * n_site)
        update_buffer = Vector{T}(undef, n_site)
        gradient_buffer = Vector{T}(undef, n_site * n_site)

        new{T}(
            gutzwiller_params,
            density_density_params,
            spin_spin_params,
            three_body_params,
            n_site,
            n_elec,
            n_spin,
            correlation_buffer,
            update_buffer,
            gradient_buffer,
            0,
            0.0,
        )
    end
end

"""
    add_gutzwiller_parameter!(jf::JastrowFactor{T}, site::Int, value::T) where T

Add a Gutzwiller parameter for a specific site.
"""
function add_gutzwiller_parameter!(jf::JastrowFactor{T}, site::Int, value::T) where {T}
    if site < 1 || site > jf.n_site
        throw(ArgumentError("Site index out of range"))
    end

    param = JastrowParameter{T}(value, GUTZWILLER, [site], Int[])
    push!(jf.gutzwiller_params, param)
    return param
end

"""
    add_density_density_parameter!(jf::JastrowFactor{T}, site1::Int, site2::Int, value::T) where T

Add a density-density correlation parameter between two sites.
"""
function add_density_density_parameter!(
    jf::JastrowFactor{T},
    site1::Int,
    site2::Int,
    value::T,
) where {T}
    if site1 < 1 || site1 > jf.n_site || site2 < 1 || site2 > jf.n_site
        throw(ArgumentError("Site index out of range"))
    end

    param = JastrowParameter{T}(value, DENSITY_DENSITY, [site1, site2], Int[])
    push!(jf.density_density_params, param)
    return param
end

"""
    add_spin_spin_parameter!(jf::JastrowFactor{T}, site1::Int, site2::Int, spin1::Int, spin2::Int, value::T) where T

Add a spin-spin correlation parameter between two sites and spins.
"""
function add_spin_spin_parameter!(
    jf::JastrowFactor{T},
    site1::Int,
    site2::Int,
    spin1::Int,
    spin2::Int,
    value::T,
) where {T}
    if site1 < 1 || site1 > jf.n_site || site2 < 1 || site2 > jf.n_site
        throw(ArgumentError("Site index out of range"))
    end
    if spin1 < 1 || spin1 > jf.n_spin || spin2 < 1 || spin2 > jf.n_spin
        throw(ArgumentError("Spin index out of range"))
    end

    param = JastrowParameter{T}(value, SPIN_SPIN, [site1, site2], [spin1, spin2])
    push!(jf.spin_spin_params, param)
    return param
end

"""
    add_three_body_parameter!(jf::JastrowFactor{T}, site1::Int, site2::Int, site3::Int, value::T) where T

Add a three-body correlation parameter between three sites.
"""
function add_three_body_parameter!(
    jf::JastrowFactor{T},
    site1::Int,
    site2::Int,
    site3::Int,
    value::T,
) where {T}
    if any(site < 1 || site > jf.n_site for site in [site1, site2, site3])
        throw(ArgumentError("Site index out of range"))
    end

    param = JastrowParameter{T}(value, THREE_BODY, [site1, site2, site3], Int[])
    push!(jf.three_body_params, param)
    return param
end

"""
    jastrow_factor(jf::JastrowFactor{T}, ele_idx::Vector{Int}, ele_cfg::Vector{Int}, ele_num::Vector{Int}) where T

Calculate the total Jastrow factor for given electron configuration.
"""
function jastrow_factor(
    jf::JastrowFactor{T},
    ele_idx::Vector{Int},
    ele_cfg::Vector{Int},
    ele_num::Vector{Int},
) where {T}
    jastrow_value = one(T)

    # Gutzwiller factors
    for param in jf.gutzwiller_params
        if param.is_active
            jastrow_value *= _calculate_gutzwiller_factor(param, ele_idx, ele_cfg, ele_num)
        end
    end

    # Density-density correlations
    for param in jf.density_density_params
        if param.is_active
            jastrow_value *=
                _calculate_density_density_factor(param, ele_idx, ele_cfg, ele_num)
        end
    end

    # Spin-spin correlations
    for param in jf.spin_spin_params
        if param.is_active
            jastrow_value *= _calculate_spin_spin_factor(param, ele_idx, ele_cfg, ele_num)
        end
    end

    # Three-body correlations
    for param in jf.three_body_params
        if param.is_active
            jastrow_value *= _calculate_three_body_factor(param, ele_idx, ele_cfg, ele_num)
        end
    end

    jf.total_evaluations += 1
    return jastrow_value
end

"""
    log_jastrow_factor(jf::JastrowFactor{T}, ele_idx::Vector{Int}, ele_cfg::Vector{Int}, ele_num::Vector{Int}) where T

Calculate the log of the total Jastrow factor.
"""
function log_jastrow_factor(
    jf::JastrowFactor{T},
    ele_idx::Vector{Int},
    ele_cfg::Vector{Int},
    ele_num::Vector{Int},
) where {T}
    log_jastrow_value = zero(T)

    # Gutzwiller factors
    for param in jf.gutzwiller_params
        if param.is_active
            log_jastrow_value +=
                _calculate_log_gutzwiller_factor(param, ele_idx, ele_cfg, ele_num)
        end
    end

    # Density-density correlations
    for param in jf.density_density_params
        if param.is_active
            log_jastrow_value +=
                _calculate_log_density_density_factor(param, ele_idx, ele_cfg, ele_num)
        end
    end

    # Spin-spin correlations
    for param in jf.spin_spin_params
        if param.is_active
            log_jastrow_value +=
                _calculate_log_spin_spin_factor(param, ele_idx, ele_cfg, ele_num)
        end
    end

    # Three-body correlations
    for param in jf.three_body_params
        if param.is_active
            log_jastrow_value +=
                _calculate_log_three_body_factor(param, ele_idx, ele_cfg, ele_num)
        end
    end

    jf.total_evaluations += 1
    return log_jastrow_value
end

"""
    jastrow_ratio(jf::JastrowFactor{T}, ele_idx_old::Vector{Int}, ele_cfg_old::Vector{Int}, ele_num_old::Vector{Int},
                  ele_idx_new::Vector{Int}, ele_cfg_new::Vector{Int}, ele_num_new::Vector{Int}) where T

Calculate the ratio of Jastrow factors for two configurations (for Metropolis sampling).
"""
function jastrow_ratio(
    jf::JastrowFactor{T},
    ele_idx_old::Vector{Int},
    ele_cfg_old::Vector{Int},
    ele_num_old::Vector{Int},
    ele_idx_new::Vector{Int},
    ele_cfg_new::Vector{Int},
    ele_num_new::Vector{Int},
) where {T}

    # Calculate log factors
    log_old = log_jastrow_factor(jf, ele_idx_old, ele_cfg_old, ele_num_old)
    log_new = log_jastrow_factor(jf, ele_idx_new, ele_cfg_new, ele_num_new)

    # Return ratio as exp(log_new - log_old)
    return exp(log_new - log_old)
end

# Helper functions for individual Jastrow factor types

function _calculate_gutzwiller_factor(
    param::JastrowParameter{T},
    ele_idx::Vector{Int},
    ele_cfg::Vector{Int},
    ele_num::Vector{Int},
) where {T}
    site = param.site_indices[1]
    if ele_cfg[site] > 0
        return exp(param.value * ele_num[site])
    else
        return one(T)
    end
end

function _calculate_log_gutzwiller_factor(
    param::JastrowParameter{T},
    ele_idx::Vector{Int},
    ele_cfg::Vector{Int},
    ele_num::Vector{Int},
) where {T}
    site = param.site_indices[1]
    if ele_cfg[site] > 0
        return param.value * ele_num[site]
    else
        return zero(T)
    end
end

function _calculate_density_density_factor(
    param::JastrowParameter{T},
    ele_idx::Vector{Int},
    ele_cfg::Vector{Int},
    ele_num::Vector{Int},
) where {T}
    site1, site2 = param.site_indices[1], param.site_indices[2]

    if ele_cfg[site1] > 0 && ele_cfg[site2] > 0
        return exp(param.value * ele_num[site1] * ele_num[site2])
    else
        return one(T)
    end
end

function _calculate_log_density_density_factor(
    param::JastrowParameter{T},
    ele_idx::Vector{Int},
    ele_cfg::Vector{Int},
    ele_num::Vector{Int},
) where {T}
    site1, site2 = param.site_indices[1], param.site_indices[2]

    if ele_cfg[site1] > 0 && ele_cfg[site2] > 0
        return param.value * ele_num[site1] * ele_num[site2]
    else
        return zero(T)
    end
end

function _calculate_spin_spin_factor(
    param::JastrowParameter{T},
    ele_idx::Vector{Int},
    ele_cfg::Vector{Int},
    ele_num::Vector{Int},
) where {T}
    site1, site2 = param.site_indices[1], param.site_indices[2]
    spin1, spin2 = param.spin_indices[1], param.spin_indices[2]

    # Simplified spin-spin correlation
    if ele_cfg[site1] > 0 && ele_cfg[site2] > 0
        # In a real implementation, this would depend on the actual spin configuration
        spin_correlation = (spin1 == spin2) ? 1.0 : -1.0
        return exp(param.value * spin_correlation * ele_num[site1] * ele_num[site2])
    else
        return one(T)
    end
end

function _calculate_log_spin_spin_factor(
    param::JastrowParameter{T},
    ele_idx::Vector{Int},
    ele_cfg::Vector{Int},
    ele_num::Vector{Int},
) where {T}
    site1, site2 = param.site_indices[1], param.site_indices[2]
    spin1, spin2 = param.spin_indices[1], param.spin_indices[2]

    if ele_cfg[site1] > 0 && ele_cfg[site2] > 0
        spin_correlation = (spin1 == spin2) ? 1.0 : -1.0
        return param.value * spin_correlation * ele_num[site1] * ele_num[site2]
    else
        return zero(T)
    end
end

function _calculate_three_body_factor(
    param::JastrowParameter{T},
    ele_idx::Vector{Int},
    ele_cfg::Vector{Int},
    ele_num::Vector{Int},
) where {T}
    site1, site2, site3 =
        param.site_indices[1], param.site_indices[2], param.site_indices[3]

    if ele_cfg[site1] > 0 && ele_cfg[site2] > 0 && ele_cfg[site3] > 0
        return exp(param.value * ele_num[site1] * ele_num[site2] * ele_num[site3])
    else
        return one(T)
    end
end

function _calculate_log_three_body_factor(
    param::JastrowParameter{T},
    ele_idx::Vector{Int},
    ele_cfg::Vector{Int},
    ele_num::Vector{Int},
) where {T}
    site1, site2, site3 =
        param.site_indices[1], param.site_indices[2], param.site_indices[3]

    if ele_cfg[site1] > 0 && ele_cfg[site2] > 0 && ele_cfg[site3] > 0
        return param.value * ele_num[site1] * ele_num[site2] * ele_num[site3]
    else
        return zero(T)
    end
end

"""
    jastrow_gradient(jf::JastrowFactor{T}, ele_idx::Vector{Int}, ele_cfg::Vector{Int}, ele_num::Vector{Int}) where T

Calculate the gradient of the log Jastrow factor with respect to all parameters.
"""
function jastrow_gradient(
    jf::JastrowFactor{T},
    ele_idx::Vector{Int},
    ele_cfg::Vector{Int},
    ele_num::Vector{Int},
) where {T}
    gradient = Vector{T}()

    # Gutzwiller gradients
    for param in jf.gutzwiller_params
        if param.is_active
            site = param.site_indices[1]
            if ele_cfg[site] > 0
                push!(gradient, ele_num[site])
            else
                push!(gradient, zero(T))
            end
        end
    end

    # Density-density gradients
    for param in jf.density_density_params
        if param.is_active
            site1, site2 = param.site_indices[1], param.site_indices[2]
            if ele_cfg[site1] > 0 && ele_cfg[site2] > 0
                push!(gradient, ele_num[site1] * ele_num[site2])
            else
                push!(gradient, zero(T))
            end
        end
    end

    # Spin-spin gradients
    for param in jf.spin_spin_params
        if param.is_active
            site1, site2 = param.site_indices[1], param.site_indices[2]
            spin1, spin2 = param.spin_indices[1], param.spin_indices[2]
            if ele_cfg[site1] > 0 && ele_cfg[site2] > 0
                spin_correlation = (spin1 == spin2) ? 1.0 : -1.0
                push!(gradient, spin_correlation * ele_num[site1] * ele_num[site2])
            else
                push!(gradient, zero(T))
            end
        end
    end

    # Three-body gradients
    for param in jf.three_body_params
        if param.is_active
            site1, site2, site3 =
                param.site_indices[1], param.site_indices[2], param.site_indices[3]
            if ele_cfg[site1] > 0 && ele_cfg[site2] > 0 && ele_cfg[site3] > 0
                push!(gradient, ele_num[site1] * ele_num[site2] * ele_num[site3])
            else
                push!(gradient, zero(T))
            end
        end
    end

    jf.total_evaluations += 1
    return gradient
end

"""
    update_jastrow_parameters!(jf::JastrowFactor{T}, gradient::Vector{T}; learning_rate::Float64 = 0.01) where T

Update Jastrow parameters using gradient descent.
"""
function update_jastrow_parameters!(
    jf::JastrowFactor{T},
    gradient::Vector{T};
    learning_rate::Float64 = 0.01,
) where {T}
    idx = 1

    # Update Gutzwiller parameters
    for param in jf.gutzwiller_params
        if param.is_active
            param.value += learning_rate * gradient[idx]
            idx += 1
        end
    end

    # Update density-density parameters
    for param in jf.density_density_params
        if param.is_active
            param.value += learning_rate * gradient[idx]
            idx += 1
        end
    end

    # Update spin-spin parameters
    for param in jf.spin_spin_params
        if param.is_active
            param.value += learning_rate * gradient[idx]
            idx += 1
        end
    end

    # Update three-body parameters
    for param in jf.three_body_params
        if param.is_active
            param.value += learning_rate * gradient[idx]
            idx += 1
        end
    end
end

"""
    get_jastrow_parameters(jf::JastrowFactor{T}) where T

Extract all Jastrow parameters as a flat vector.
"""
function get_jastrow_parameters(jf::JastrowFactor{T}) where {T}
    params = Vector{T}()

    for param in jf.gutzwiller_params
        if param.is_active
            push!(params, param.value)
        end
    end

    for param in jf.density_density_params
        if param.is_active
            push!(params, param.value)
        end
    end

    for param in jf.spin_spin_params
        if param.is_active
            push!(params, param.value)
        end
    end

    for param in jf.three_body_params
        if param.is_active
            push!(params, param.value)
        end
    end

    return params
end

"""
    set_jastrow_parameters!(jf::JastrowFactor{T}, params::Vector{T}) where T

Set Jastrow parameters from a flat vector.
"""
function set_jastrow_parameters!(jf::JastrowFactor{T}, params::Vector{T}) where {T}
    idx = 1

    for param in jf.gutzwiller_params
        if param.is_active
            param.value = params[idx]
            idx += 1
        end
    end

    for param in jf.density_density_params
        if param.is_active
            param.value = params[idx]
            idx += 1
        end
    end

    for param in jf.spin_spin_params
        if param.is_active
            param.value = params[idx]
            idx += 1
        end
    end

    for param in jf.three_body_params
        if param.is_active
            param.value = params[idx]
            idx += 1
        end
    end
end

"""
    jastrow_parameter_count(jf::JastrowFactor{T}) where T

Get total number of active Jastrow parameters.
"""
function jastrow_parameter_count(jf::JastrowFactor{T}) where {T}
    count = 0

    count += sum(param.is_active for param in jf.gutzwiller_params; init = 0)
    count += sum(param.is_active for param in jf.density_density_params; init = 0)
    count += sum(param.is_active for param in jf.spin_spin_params; init = 0)
    count += sum(param.is_active for param in jf.three_body_params; init = 0)

    return count
end

"""
    reset_jastrow!(jf::JastrowFactor{T}) where T

Reset Jastrow factor to initial state.
"""
function reset_jastrow!(jf::JastrowFactor{T}) where {T}
    for param in jf.gutzwiller_params
        param.value = zero(T)
    end

    for param in jf.density_density_params
        param.value = zero(T)
    end

    for param in jf.spin_spin_params
        param.value = zero(T)
    end

    for param in jf.three_body_params
        param.value = zero(T)
    end

    jf.total_evaluations = 0
    jf.evaluation_time = 0.0
end

"""
    benchmark_jastrow(n_site::Int = 10, n_elec::Int = 5, n_iterations::Int = 1000)

Benchmark Jastrow factor calculations.
"""
function benchmark_jastrow(n_site::Int = 10, n_elec::Int = 5, n_iterations::Int = 1000)
    println(
        "Benchmarking Jastrow factors (n_site=$n_site, n_elec=$n_elec, iterations=$n_iterations)...",
    )

    # Create Jastrow factor
    jf = JastrowFactor{ComplexF64}(n_site, n_elec)

    # Add some parameters
    for i = 1:n_site
        add_gutzwiller_parameter!(jf, i, ComplexF64(0.1))
    end

    for i = 1:n_site-1
        add_density_density_parameter!(jf, i, i + 1, ComplexF64(0.05))
    end

    # Initialize electron configuration
    ele_idx = collect(1:n_elec)
    ele_cfg = zeros(Int, n_site)
    ele_cfg[1:n_elec] .= 1
    ele_num = copy(ele_cfg)

    # Benchmark Jastrow factor calculation
    @time begin
        for _ = 1:n_iterations
            jastrow_factor(jf, ele_idx, ele_cfg, ele_num)
        end
    end
    println("  Jastrow factor calculation rate")

    # Benchmark gradient calculation
    @time begin
        for _ = 1:n_iterations
            jastrow_gradient(jf, ele_idx, ele_cfg, ele_num)
        end
    end
    println("  Jastrow gradient calculation rate")

    println("Jastrow benchmark completed.")
    println("  Total evaluations: $(jf.total_evaluations)")
end
