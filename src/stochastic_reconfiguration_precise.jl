"""
Precise Stochastic Reconfiguration

High-precision implementation of stochastic reconfiguration method
following the C reference implementation in mVMC.
"""

using LinearAlgebra
using Printf
using SparseArrays

"""
    PreciseStochasticReconfiguration{T}

Enhanced stochastic reconfiguration optimizer with features:
- Multiple regularization schemes
- Conjugate gradient solver option
- Numerical stability improvements
- Energy variance optimization
"""
mutable struct PreciseStochasticReconfiguration{T<:Union{Float64,ComplexF64}}
    n_parameters::Int
    n_samples::Int

    # Overlap matrix S and force vector F
    overlap_matrix::Matrix{T}
    force_vector::Vector{T}
    parameter_delta::Vector{T}

    # Gradient information
    gradients::Matrix{T}  # n_parameters × n_samples
    energy_gradients::Matrix{T}

    # Energy and weights
    energy_samples::Vector{T}
    sample_weights::Vector{Float64}

    # Regularization parameters
    regularization::Float64
    diagonal_shift::Float64
    eigenvalue_cutoff::Float64

    # C-compatible stabilization parameters
    dsr_opt_red_cut::Float64  # DSROptRedCut: threshold for redundant direction truncation
    dsr_opt_sta_del::Float64  # DSROptStaDel: diagonal stabilization factor
    parameter_mask::Vector{Bool}  # Which parameters to optimize (after redundancy check)

    # Conjugate gradient parameters
    use_cg::Bool
    cg_max_iterations::Int
    cg_tolerance::Float64
    cg_residual::Vector{T}
    cg_direction::Vector{T}
    cg_temp::Vector{T}

    # Numerical stability
    use_svd::Bool
    condition_number_threshold::Float64
    max_parameter_change::Float64

    # Statistics tracking
    overlap_eigenvalues::Vector{Float64}
    overlap_condition_number::Float64
    optimization_step_size::Float64

    # Temporary arrays for efficiency
    temp_vector::Vector{T}
    temp_matrix::Matrix{T}

    # Weighted accumulators (OO and HO) for C-compatible SR stats
    wc::Float64
    ho_accum::Vector{T}
    oo_accum::Matrix{T}

    function PreciseStochasticReconfiguration{T}(n_parameters::Int, n_samples::Int) where {T}
        new{T}(
            n_parameters,                           # n_parameters
            n_samples,                              # n_samples
            zeros(T, n_parameters, n_parameters),   # overlap_matrix
            zeros(T, n_parameters),                 # force_vector
            zeros(T, n_parameters),                 # parameter_delta
            zeros(T, n_parameters, n_samples),      # gradients
            zeros(T, n_parameters, n_samples),      # energy_gradients
            zeros(T, n_samples),                    # energy_samples
            ones(Float64, n_samples),               # sample_weights
            1e-4,                                   # regularization
            1e-3,                                   # diagonal_shift
            1e-10,                                  # eigenvalue_cutoff
            0.001,                                  # dsr_opt_red_cut (DSROptRedCut)
            0.01,                                   # dsr_opt_sta_del (DSROptStaDel)
            trues(n_parameters),                    # parameter_mask
            false,                                  # use_cg
            100,                                    # cg_max_iterations
            1e-6,                                   # cg_tolerance
            zeros(T, n_parameters),                 # cg_residual
            zeros(T, n_parameters),                 # cg_direction
            zeros(T, n_parameters),                 # cg_temp
            false,                                  # use_svd
            1e12,                                   # condition_number_threshold
            0.1,                                    # max_parameter_change
            zeros(Float64, n_parameters),           # overlap_eigenvalues
            1.0,                                    # overlap_condition_number
            0.02,                                   # optimization_step_size
            zeros(T, n_parameters),                 # temp_vector
            zeros(T, n_parameters, n_parameters),   # temp_matrix
            0.0,                                    # wc
            zeros(T, n_parameters),                 # ho_accum
            zeros(T, n_parameters, n_parameters)    # oo_accum
        )
    end
end

PreciseStochasticReconfiguration(n_parameters::Int, n_samples::Int; T=ComplexF64) =
    PreciseStochasticReconfiguration{T}(n_parameters, n_samples)

"""
    configure_optimization!(sr::PreciseStochasticReconfiguration{T}, config::OptimizationConfig;
                           dsr_opt_red_cut=nothing, dsr_opt_sta_del=nothing) where {T}

Configure the stochastic reconfiguration optimizer with optimization settings.
Optionally override C-compatible stabilization parameters.
"""
function configure_optimization!(sr::PreciseStochasticReconfiguration{T}, config::OptimizationConfig;
                                dsr_opt_red_cut=nothing, dsr_opt_sta_del=nothing) where {T}
    sr.regularization = config.regularization_parameter
    sr.optimization_step_size = config.learning_rate
    sr.diagonal_shift = config.regularization_parameter * 10
    sr.use_cg = config.use_sr_cg
    sr.cg_max_iterations = config.sr_cg_max_iter
    sr.cg_tolerance = config.sr_cg_tol

    # Override C-compatible parameters if provided
    if dsr_opt_red_cut !== nothing
        sr.dsr_opt_red_cut = Float64(dsr_opt_red_cut)
    end
    if dsr_opt_sta_del !== nothing
        sr.dsr_opt_sta_del = Float64(dsr_opt_sta_del)
    end

    # Resize arrays if necessary
    if size(sr.overlap_matrix, 1) != sr.n_parameters
        sr.overlap_matrix = zeros(T, sr.n_parameters, sr.n_parameters)
        sr.force_vector = zeros(T, sr.n_parameters)
        sr.parameter_delta = zeros(T, sr.n_parameters)
        sr.temp_vector = zeros(T, sr.n_parameters)
        sr.temp_matrix = zeros(T, sr.n_parameters, sr.n_parameters)
        sr.cg_residual = zeros(T, sr.n_parameters)
        sr.cg_direction = zeros(T, sr.n_parameters)
        sr.cg_temp = zeros(T, sr.n_parameters)
    end

    if size(sr.gradients, 2) != sr.n_samples
        sr.gradients = zeros(T, sr.n_parameters, sr.n_samples)
        sr.energy_gradients = zeros(T, sr.n_parameters, sr.n_samples)
        sr.energy_samples = zeros(T, sr.n_samples)
        sr.sample_weights = ones(Float64, sr.n_samples)
    end

    # Reset weighted accumulators each (re)configure
    sr.wc = 0.0
    sr.ho_accum .= zero(T)
    fill!(sr.oo_accum, zero(T))
end

"""
    compute_overlap_matrix_precise!(sr::PreciseStochasticReconfiguration{T}, gradients::Matrix{T}, weights::Vector{Float64}) where {T}

Compute overlap matrix S_{ij} = ⟨O_i* O_j⟩ - ⟨O_i*⟩⟨O_j⟩ with high precision.
"""
function compute_overlap_matrix_precise!(sr::PreciseStochasticReconfiguration{T}, gradients::Matrix{T}, weights::Vector{Float64}) where {T}
    n_params, n_samples = size(gradients)
    @assert n_params == sr.n_parameters
    @assert n_samples == sr.n_samples

    # Store gradients
    sr.gradients .= gradients
    sr.sample_weights .= weights

    # Normalize weights
    weight_sum = sum(weights)
    if weight_sum > 0
        sr.sample_weights ./= weight_sum
    else
        sr.sample_weights .= 1.0 / n_samples
    end

    # Compute average gradients ⟨O_i⟩
    fill!(sr.temp_vector, zero(T))
    for i in 1:n_params
        for k in 1:n_samples
            sr.temp_vector[i] += sr.sample_weights[k] * conj(gradients[i, k])
        end
    end

    # Compute overlap matrix S_{ij} = ⟨O_i* O_j⟩ - ⟨O_i*⟩⟨O_j⟩
    fill!(sr.overlap_matrix, zero(T))

    # First compute ⟨O_i* O_j⟩
    for i in 1:n_params
        for j in 1:n_params
            overlap_ij = zero(T)
            for k in 1:n_samples
                overlap_ij += sr.sample_weights[k] * conj(gradients[i, k]) * gradients[j, k]
            end
            sr.overlap_matrix[i, j] = overlap_ij
        end
    end

    # Subtract ⟨O_i*⟩⟨O_j⟩
    for i in 1:n_params
        for j in 1:n_params
            sr.overlap_matrix[i, j] -= conj(sr.temp_vector[i]) * sr.temp_vector[j]
        end
    end

    # Add regularization
    regularize_overlap_matrix!(sr)

    # Compute eigenvalues for diagnostics
    compute_overlap_eigenvalues!(sr)
end

"""
    apply_redundant_direction_truncation!(sr::PreciseStochasticReconfiguration{T}) where {T}

Apply C-compatible redundant direction truncation based on diagonal elements.
This implements the DSROptRedCut mechanism from mVMC stcopt.c:
- Compute diagonal elements S[i][i] = ⟨O_i* O_i⟩ - ⟨O_i*⟩²
- Find max diagonal: sDiagMax
- Set threshold: diagCutThreshold = sDiagMax * DSROptRedCut
- Disable parameters with diagonal < threshold
"""
function apply_redundant_direction_truncation!(sr::PreciseStochasticReconfiguration{T}) where {T}
    # Compute diagonal variances (before regularization)
    diag_variances = zeros(Float64, sr.n_parameters)
    for i in 1:sr.n_parameters
        diag_variances[i] = real(sr.overlap_matrix[i, i])
    end

    # Find max diagonal variance
    max_diag = maximum(diag_variances)

    # Apply threshold: disable parameters with variance < max * DSROptRedCut
    threshold = max_diag * sr.dsr_opt_red_cut
    n_cut = 0
    n_opt = 0

    for i in 1:sr.n_parameters
        if diag_variances[i] < threshold
            sr.parameter_mask[i] = false  # Disable this parameter
            n_cut += 1
        else
            sr.parameter_mask[i] = true   # Enable this parameter
            n_opt += 1
        end
    end

    # Report (for debugging)
    if n_cut > 0
        # println("  SR: Truncated $n_cut redundant directions (threshold=$(threshold), max_diag=$(max_diag))")
    end

    return n_opt, n_cut
end

"""
    regularize_overlap_matrix!(sr::PreciseStochasticReconfiguration{T}) where {T}

Apply regularization to the overlap matrix for numerical stability.
Implements both standard regularization and C-compatible DSROptStaDel.
"""
function regularize_overlap_matrix!(sr::PreciseStochasticReconfiguration{T}) where {T}
    # First apply redundant direction truncation (C-compatible)
    apply_redundant_direction_truncation!(sr)

    # C-compatible diagonal stabilization (DSROptStaDel)
    # Add DSROptStaDel * max(diagonal) to all diagonal elements
    diag_values = [real(sr.overlap_matrix[i, i]) for i in 1:sr.n_parameters]
    max_diag = maximum(abs.(diag_values))
    sta_del_value = sr.dsr_opt_sta_del * max_diag

    for i in 1:sr.n_parameters
        sr.overlap_matrix[i, i] += T(sta_del_value)
    end

    # Additional diagonal shift regularization (original implementation)
    for i in 1:sr.n_parameters
        sr.overlap_matrix[i, i] += T(sr.diagonal_shift)
    end

    # Additional regularization based on matrix norm
    matrix_norm = norm(sr.overlap_matrix)
    if matrix_norm > 0
        regularization_strength = sr.regularization * matrix_norm / sr.n_parameters
        for i in 1:sr.n_parameters
            sr.overlap_matrix[i, i] += T(regularization_strength)
        end
    end

    # Zero out rows/columns for masked parameters
    for i in 1:sr.n_parameters
        if !sr.parameter_mask[i]
            sr.overlap_matrix[i, :] .= zero(T)
            sr.overlap_matrix[:, i] .= zero(T)
            sr.overlap_matrix[i, i] = one(T)  # Set diagonal to 1 to avoid singularity
        end
    end
end

"""
    compute_overlap_eigenvalues!(sr::PreciseStochasticReconfiguration{T}) where {T}

Compute eigenvalues of overlap matrix for condition number analysis.
"""
function compute_overlap_eigenvalues!(sr::PreciseStochasticReconfiguration{T}) where {T}
    try
        if T <: Real
            sr.overlap_eigenvalues .= eigvals(Hermitian(sr.overlap_matrix))
        else
            eigenvals = eigvals(sr.overlap_matrix)
            sr.overlap_eigenvalues .= real(eigenvals)
        end

        # Compute condition number
        min_eigenval = minimum(sr.overlap_eigenvalues)
        max_eigenval = maximum(sr.overlap_eigenvalues)

        if min_eigenval > 0
            sr.overlap_condition_number = max_eigenval / min_eigenval
        else
            sr.overlap_condition_number = Inf
        end
    catch
        # Fallback if eigenvalue computation fails
        sr.overlap_condition_number = 1e12
        fill!(sr.overlap_eigenvalues, 1.0)
    end
end

"""
    compute_force_vector_precise!(sr::PreciseStochasticReconfiguration{T}, gradients::Matrix{T}, energies::Vector{T}, weights::Vector{Float64}) where {T}

Compute force vector F_i = ⟨O_i* H⟩ - ⟨O_i*⟩⟨H⟩ with high precision.
"""
function compute_force_vector_precise!(sr::PreciseStochasticReconfiguration{T}, gradients::Matrix{T}, energies::Vector{T}, weights::Vector{Float64}) where {T}
    n_params, n_samples = size(gradients)
    @assert length(energies) == n_samples
    @assert length(weights) == n_samples

    # Store energy samples
    sr.energy_samples .= energies

    # Compute weighted average energy ⟨H⟩ (also accumulate OO/HO like C's vmccal.c)
    weighted_energy = zero(T)
    for k in 1:n_samples
        weighted_energy += sr.sample_weights[k] * energies[k]
    end

    # Accumulate OO and HO moments with the same normalized weights
    # HO_i = ⟨O_i* H⟩_w, OO_ij = ⟨O_i* O_j⟩_w
    fill!(sr.ho_accum, zero(T))
    fill!(sr.oo_accum, zero(T))
    for k in 1:n_samples
        wk = T(sr.sample_weights[k])
        ek = T(energies[k])
        gk = view(gradients, :, k)
        # HO
        sr.ho_accum .+= wk .* conj.(gk) .* ek
        # OO rank-1
        @inbounds for i in 1:n_params
            ci = wk * conj(gk[i])
            @inbounds for j in 1:n_params
                sr.oo_accum[i, j] += ci * gk[j]
            end
        end
    end

    # Compute average gradients ⟨O_i*⟩ (already computed in overlap matrix)
    fill!(sr.temp_vector, zero(T))
    for i in 1:n_params
        for k in 1:n_samples
            sr.temp_vector[i] += sr.sample_weights[k] * conj(gradients[i, k])
        end
    end

    # Compute force vector F_i = ⟨O_i* H⟩ - ⟨O_i*⟩⟨H⟩
    fill!(sr.force_vector, zero(T))

    # First compute ⟨O_i* H⟩
    for i in 1:n_params
        force_i = zero(T)
        for k in 1:n_samples
            force_i += sr.sample_weights[k] * conj(gradients[i, k]) * energies[k]
        end
        sr.force_vector[i] = force_i
    end

    # Subtract ⟨O_i*⟩⟨H⟩
    for i in 1:n_params
        sr.force_vector[i] -= conj(sr.temp_vector[i]) * weighted_energy
    end

    # Apply parameter mask (zero out force for masked parameters)
    for i in 1:n_params
        if !sr.parameter_mask[i]
            sr.force_vector[i] = zero(T)
        end
    end
end

"""
    solve_sr_equations_direct!(sr::PreciseStochasticReconfiguration{T}) where {T}

Solve S δp = -η F using direct matrix inversion with SVD for stability.
"""
function solve_sr_equations_direct!(sr::PreciseStochasticReconfiguration{T}) where {T}
    try
        if sr.use_svd || sr.overlap_condition_number > sr.condition_number_threshold
            # Use SVD for better numerical stability
            solve_with_svd!(sr)
        else
            # Use LU decomposition (faster)
            solve_with_lu!(sr)
        end

        # Apply step size
        sr.parameter_delta .*= T(-sr.optimization_step_size)

        # Apply parameter mask (zero out updates for masked parameters)
        for i in 1:sr.n_parameters
            if !sr.parameter_mask[i]
                sr.parameter_delta[i] = zero(T)
            end
        end

        # Limit parameter changes for stability
        limit_parameter_changes!(sr)

    catch e
        @warn "SR equation solve failed: $e, using fallback"
        # Fallback: simple gradient descent
        sr.parameter_delta .= T(-sr.optimization_step_size * 0.1) .* sr.force_vector
    end
end

"""
    solve_with_svd!(sr::PreciseStochasticReconfiguration{T}) where {T}

Solve using SVD with eigenvalue cutoff.
"""
function solve_with_svd!(sr::PreciseStochasticReconfiguration{T}) where {T}
    # Compute SVD of overlap matrix
    if T <: Real
        U, σ, Vt = svd(sr.overlap_matrix)
        V = Vt'
    else
        U, σ, V = svd(sr.overlap_matrix)
    end

    # Apply eigenvalue cutoff
    σ_inv = similar(σ)
    for i in eachindex(σ)
        if σ[i] > sr.eigenvalue_cutoff
            σ_inv[i] = 1.0 / σ[i]
        else
            σ_inv[i] = 0.0
        end
    end

    # Compute pseudoinverse: S^(-1) = V Σ^(-1) U*
    if T <: Real
        sr.parameter_delta .= V * (σ_inv .* (U' * sr.force_vector))
    else
        sr.parameter_delta .= V * (σ_inv .* (U' * sr.force_vector))
    end
end

"""
    solve_with_lu!(sr::PreciseStochasticReconfiguration{T}) where {T}

Solve using LU decomposition.
"""
function solve_with_lu!(sr::PreciseStochasticReconfiguration{T}) where {T}
    # Try Cholesky first (if matrix is positive definite)
    try
        if T <: Real
            chol = cholesky(Hermitian(sr.overlap_matrix))
            sr.parameter_delta .= chol \ sr.force_vector
        else
            # For complex case, use LU
            sr.parameter_delta .= sr.overlap_matrix \ sr.force_vector
        end
    catch
        # Fallback to general LU
        sr.parameter_delta .= sr.overlap_matrix \ sr.force_vector
    end
end

"""
    solve_sr_equations_cg!(sr::PreciseStochasticReconfiguration{T}) where {T}

Solve S δp = -η F using conjugate gradient method.
"""
function solve_sr_equations_cg!(sr::PreciseStochasticReconfiguration{T}) where {T}
    # Initialize
    rhs = T(-sr.optimization_step_size) .* sr.force_vector
    fill!(sr.parameter_delta, zero(T))

    # Initial residual: r = b - A*x = b (since x = 0)
    sr.cg_residual .= rhs
    sr.cg_direction .= sr.cg_residual

    # Initial residual norm
    residual_norm_sq = real(dot(sr.cg_residual, sr.cg_residual))
    initial_residual_norm = sqrt(residual_norm_sq)

    # Tolerance check
    if initial_residual_norm < sr.cg_tolerance
        return
    end

    # CG iterations
    for iter in 1:sr.cg_max_iterations
        # Compute A * direction
        mul!(sr.cg_temp, sr.overlap_matrix, sr.cg_direction)

        # Compute step size: α = (r^T r) / (p^T A p)
        denominator = real(dot(sr.cg_direction, sr.cg_temp))
        if abs(denominator) < 1e-15
            break
        end
        alpha = residual_norm_sq / denominator

        # Update solution: x = x + α * p
        sr.parameter_delta .+= T(alpha) .* sr.cg_direction

        # Update residual: r = r - α * A * p
        sr.cg_residual .-= T(alpha) .* sr.cg_temp

        # Check convergence
        new_residual_norm_sq = real(dot(sr.cg_residual, sr.cg_residual))
        relative_residual = sqrt(new_residual_norm_sq) / initial_residual_norm

        if relative_residual < sr.cg_tolerance
            break
        end

        # Compute new direction: β = (r_new^T r_new) / (r_old^T r_old)
        beta = new_residual_norm_sq / residual_norm_sq
        residual_norm_sq = new_residual_norm_sq

        # Update direction: p = r + β * p
        sr.cg_direction .= sr.cg_residual .+ T(beta) .* sr.cg_direction
    end

    # Limit parameter changes
    limit_parameter_changes!(sr)
end

"""
    limit_parameter_changes!(sr::PreciseStochasticReconfiguration{T}) where {T}

Limit parameter changes to prevent numerical instability.
"""
function limit_parameter_changes!(sr::PreciseStochasticReconfiguration{T}) where {T}
    # Compute maximum change
    max_change = maximum(abs.(sr.parameter_delta))

    # Apply limit if necessary
    if max_change > sr.max_parameter_change
        scaling_factor = sr.max_parameter_change / max_change
        sr.parameter_delta .*= T(scaling_factor)
    end
end

"""
    solve_sr_equations_precise!(sr::PreciseStochasticReconfiguration{T}, config::OptimizationConfig) where {T}

Main interface for solving SR equations with method selection.
"""
function solve_sr_equations_precise!(sr::PreciseStochasticReconfiguration{T}, config::OptimizationConfig) where {T}
    if sr.use_cg
        solve_sr_equations_cg!(sr)
    else
        solve_sr_equations_direct!(sr)
    end
end

"""
    compute_energy_variance!(sr::PreciseStochasticReconfiguration{T}) -> Float64

Compute energy variance for monitoring optimization quality.
"""
function compute_energy_variance!(sr::PreciseStochasticReconfiguration{T}) where {T}
    # Weighted average energy
    avg_energy = zero(T)
    for k in 1:sr.n_samples
        avg_energy += sr.sample_weights[k] * sr.energy_samples[k]
    end

    # Weighted energy variance
    variance = 0.0
    for k in 1:sr.n_samples
        diff = sr.energy_samples[k] - avg_energy
        variance += sr.sample_weights[k] * real(diff * conj(diff))
    end

    return variance
end

"""
    print_sr_diagnostics(sr::PreciseStochasticReconfiguration{T}, iteration::Int) where {T}

Print diagnostic information about the SR optimization.
"""
function print_sr_diagnostics(sr::PreciseStochasticReconfiguration{T}, iteration::Int) where {T}
    @printf("SR Diagnostics [iter %d]:\n", iteration)
    @printf("  Condition number: %.2e\n", sr.overlap_condition_number)
    @printf("  Min eigenvalue:   %.2e\n", minimum(sr.overlap_eigenvalues))
    @printf("  Max eigenvalue:   %.2e\n", maximum(sr.overlap_eigenvalues))
    @printf("  Force norm:       %.2e\n", norm(sr.force_vector))
    @printf("  Parameter change: %.2e\n", norm(sr.parameter_delta))
    @printf("  Energy variance:  %.2e\n", compute_energy_variance!(sr))
    println()
end
