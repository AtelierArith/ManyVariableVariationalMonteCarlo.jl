"""
Stochastic Reconfiguration (SR) Matrix Calculation

Calculates S matrix and force vector for parameter optimization.

S[i,j] = <O_i* O_j> - <O_i*><O_j>
g[i] = <O_i* E_local> - <O_i*><E_local>

where O_i = ∂log(ψ)/∂α_i

Based on mVMC C implementation:
- sr_calgrn.c: SR matrix calculation
- stcopt.c: Optimization using SR
"""

using LinearAlgebra
using Statistics

"""
    SRMatrixCalculator{T}

Calculator for SR matrix and force vector.
"""
mutable struct SRMatrixCalculator{T<:Number}
    n_para::Int  # Number of parameters

    # Accumulated values over samples
    sum_o::Vector{T}          # Σ O_i
    sum_o_conj::Vector{T}     # Σ O_i*
    sum_oo::Matrix{T}         # Σ O_i* O_j
    sum_e::T                  # Σ E_local
    sum_oe::Vector{T}         # Σ O_i* E_local

    n_samples::Int            # Number of samples accumulated

    # Results
    s_matrix::Matrix{T}       # S matrix
    force_vector::Vector{T}   # Force vector

    function SRMatrixCalculator{T}(n_para::Int) where {T}
        new{T}(
            n_para,
            zeros(T, n_para),
            zeros(T, n_para),
            zeros(T, n_para, n_para),
            zero(T),
            zeros(T, n_para),
            0,
            zeros(T, n_para, n_para),
            zeros(T, n_para),
        )
    end
end

"""
    reset!(calc::SRMatrixCalculator)

Reset accumulated values.

C実装参考: sr_calgrn.c 1行目から500行目まで
"""
function reset!(calc::SRMatrixCalculator)
    fill!(calc.sum_o, 0)
    fill!(calc.sum_o_conj, 0)
    fill!(calc.sum_oo, 0)
    calc.sum_e = 0
    fill!(calc.sum_oe, 0)
    calc.n_samples = 0
    return nothing
end

"""
    calculate_o_derivatives(
        parameters::Vector{T},
        state::FSZSamplerState{T}
    ) where T

Calculate O_i = ∂log(ψ)/∂α_i for current configuration.

For simple variational wavefunctions:
- Gutzwiller: O_i = occupation at site i
- Jastrow: O_i = pair correlations
- Orbital: O_i = orbital overlap derivatives

Simplified implementation returns mock derivatives.
"""
function calculate_o_derivatives(
    parameters::Vector{T},
    state::FSZSamplerState{T},
) where {T}
    n_para = length(parameters)
    o_derivatives = zeros(T, n_para)

    # For Gutzwiller-type parameters (first n_sites parameters)
    n_sites = state.config.n_sites
    if n_para >= n_sites
        for i in 1:n_sites
            # Derivative w.r.t. Gutzwiller parameter at site i
            # Proportional to double occupancy at site i
            i_up = state.ele_cfg[i]
            i_down = state.ele_cfg[i + n_sites]
            if i_up >= 0 && i_down >= 0
                o_derivatives[i] = 1.0
            end
        end
    end

    # For orbital/Jastrow parameters (remaining parameters)
    # Use simplified derivatives
    for i in (n_sites+1):n_para
        o_derivatives[i] = randn() * 0.1  # Placeholder
    end

    return o_derivatives
end

"""
    accumulate_sample!(
        calc::SRMatrixCalculator{T},
        o_derivatives::Vector{T},
        energy::T
    ) where T

Accumulate one sample for SR matrix calculation.
"""
function accumulate_sample!(
    calc::SRMatrixCalculator{T},
    o_derivatives::Vector{T},
    energy::T,
) where {T}
    n_para = calc.n_para

    # Accumulate O values
    for i in 1:n_para
        calc.sum_o[i] += o_derivatives[i]
        calc.sum_o_conj[i] += conj(o_derivatives[i])
    end

    # Accumulate O* O products
    for i in 1:n_para
        for j in 1:n_para
            calc.sum_oo[i, j] += conj(o_derivatives[i]) * o_derivatives[j]
        end
    end

    # Accumulate energy
    calc.sum_e += energy

    # Accumulate O* E products
    for i in 1:n_para
        calc.sum_oe[i] += conj(o_derivatives[i]) * energy
    end

    calc.n_samples += 1

    return nothing
end

"""
    finalize_calculation!(calc::SRMatrixCalculator{T}) where T

Finalize SR matrix and force vector calculation.

S[i,j] = <O_i* O_j> - <O_i*><O_j>
g[i] = <O_i* E> - <O_i*><E>
"""
function finalize_calculation!(calc::SRMatrixCalculator{T}) where {T}
    if calc.n_samples == 0
        @warn "No samples accumulated for SR calculation"
        return nothing
    end

    n_para = calc.n_para
    n = calc.n_samples

    # Calculate averages
    avg_o = calc.sum_o / n
    avg_o_conj = calc.sum_o_conj / n
    avg_oo = calc.sum_oo / n
    avg_e = calc.sum_e / n
    avg_oe = calc.sum_oe / n

    # Calculate S matrix: <O_i* O_j> - <O_i*><O_j>
    for i in 1:n_para
        for j in 1:n_para
            calc.s_matrix[i, j] = avg_oo[i, j] - avg_o_conj[i] * avg_o[j]
        end
    end

    # Calculate force vector: <O_i* E> - <O_i*><E>
    for i in 1:n_para
        calc.force_vector[i] = avg_oe[i] - avg_o_conj[i] * avg_e
    end

    return nothing
end

"""
    solve_sr_equations(
        s_matrix::Matrix{T},
        force_vector::Vector{T};
        stabilization::Float64 = 1e-5,
        cutoff::Float64 = 1e-10
    ) where T

Solve SR equations: S * dx = -g

# Arguments
- `s_matrix`: S matrix
- `force_vector`: Force vector g
- `stabilization`: Diagonal shift for stabilization
- `cutoff`: Eigenvalue cutoff for ill-conditioned matrix

# Returns
- `dx`: Parameter update direction
"""
function solve_sr_equations(
    s_matrix::Matrix{T},
    force_vector::Vector{T};
    stabilization::Float64 = 1e-5,
    cutoff::Float64 = 1e-10,
) where {T}
    n_para = length(force_vector)

    # Add diagonal stabilization
    s_stabilized = copy(s_matrix)
    for i in 1:n_para
        s_stabilized[i, i] += stabilization
    end

    # Eigenvalue decomposition for regularization
    try
        eigen_decomp = eigen(Hermitian(s_stabilized))
        eigenvalues = eigen_decomp.values
        eigenvectors = eigen_decomp.vectors

        # Filter small eigenvalues
        n_keep = count(λ -> abs(λ) > cutoff, eigenvalues)

        if n_keep == 0
            @warn "All eigenvalues below cutoff, returning zero update"
            return zeros(T, n_para)
        end

        # Reconstruct inverse with filtered eigenvalues
        s_inv = zeros(T, n_para, n_para)
        for i in 1:n_para
            if abs(eigenvalues[i]) > cutoff
                for j in 1:n_para
                    for k in 1:n_para
                        s_inv[j, k] += eigenvectors[j, i] * conj(eigenvectors[k, i]) / eigenvalues[i]
                    end
                end
            end
        end

        # Solve: dx = -S^{-1} * g
        dx = -s_inv * force_vector

        return dx

    catch e
        @warn "Failed to solve SR equations: $e"
        return zeros(T, n_para)
    end
end

"""
    run_sr_optimization_step(
        calc::SRMatrixCalculator{T},
        parameters::Vector{T},
        states::Vector{FSZSamplerState{T}},
        energies::Vector{T};
        step_size::Float64 = 0.01,
        stabilization::Float64 = 1e-5
    ) where T

Run one step of SR optimization.

# Returns
- `dx`: Parameter update
- `avg_energy`: Average energy
"""
function run_sr_optimization_step(
    calc::SRMatrixCalculator{T},
    parameters::Vector{T},
    states::Vector{FSZSamplerState{T}},
    energies::Vector{T};
    step_size::Float64 = 0.01,
    stabilization::Float64 = 1e-5,
) where {T}
    # Reset calculator
    reset!(calc)

    # Accumulate samples
    for (state, energy) in zip(states, energies)
        o_derivatives = calculate_o_derivatives(parameters, state)
        accumulate_sample!(calc, o_derivatives, energy)
    end

    # Finalize calculation
    finalize_calculation!(calc)

    # Solve SR equations
    dx = solve_sr_equations(
        calc.s_matrix,
        calc.force_vector;
        stabilization = stabilization,
    )

    # Apply step size
    dx .*= step_size

    # Calculate average energy
    avg_energy = sum(energies) / length(energies)

    return (dx, avg_energy)
end

"""
    print_sr_matrix_info(calc::SRMatrixCalculator)

Print SR matrix calculation info.
"""
function print_sr_matrix_info(calc::SRMatrixCalculator)
    println("SR Matrix Calculator:")
    println("  n_para: $(calc.n_para)")
    println("  n_samples: $(calc.n_samples)")

    if calc.n_samples > 0
        println("  S matrix condition number: $(cond(calc.s_matrix))")
        println("  Force vector norm: $(norm(calc.force_vector))")
    end
end
