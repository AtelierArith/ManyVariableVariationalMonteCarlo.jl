"""
Slater determinant implementation for ManyVariableVariationalMonteCarlo.jl

Implements Slater determinant wavefunction components including:
- Determinant evaluation and updates
- Fast ratio calculations for Metropolis sampling
- Frozen-spin variants
- Backflow corrections

Ported from slater.c and slater_fsz.c in the C reference implementation.
"""

using LinearAlgebra
using LinearAlgebra.BLAS
using LinearAlgebra.LAPACK

"""
    SlaterMatrix{T}

Represents a Slater determinant matrix with associated metadata.
"""
mutable struct SlaterMatrix{T<:Union{Float64,ComplexF64}}
    matrix::Matrix{T}
    n_elec::Int
    n_orb::Int
    det_value::T
    log_det_value::Float64
    is_valid::Bool
end

function SlaterMatrix{T}(n_elec::Int, n_orb::Int) where {T<:Union{Float64,ComplexF64}}
    matrix = zeros(T, n_elec, n_orb)
    return SlaterMatrix{T}(matrix, n_elec, n_orb, zero(T), 0.0, false)
end

"""
    SlaterDeterminant{T}

Main Slater determinant structure containing all necessary components.
"""
mutable struct SlaterDeterminant{T<:Union{Float64,ComplexF64}}
    # Core matrices
    slater_matrix::SlaterMatrix{T}
    inverse_matrix::Matrix{T}

    # Orbital information
    orbital_indices::Vector{Int}
    orbital_signs::Vector{Int}

    # Update tracking
    update_count::Int
    last_update_row::Int
    last_update_col::Int

    # Performance optimization
    workspace::Vector{T}
    pivot_indices::Vector{Int}

    function SlaterDeterminant{T}(
        n_elec::Int,
        n_orb::Int,
    ) where {T<:Union{Float64,ComplexF64}}
        slater_matrix = SlaterMatrix{T}(n_elec, n_orb)
        inverse_matrix = zeros(T, n_elec, n_elec)
        orbital_indices = zeros(Int, n_elec)
        orbital_signs = ones(Int, n_elec)
        workspace = Vector{T}(undef, max(n_elec, n_orb))
        pivot_indices = zeros(Int, n_elec)

        new{T}(
            slater_matrix,
            inverse_matrix,
            orbital_indices,
            orbital_signs,
            0,
            -1,
            -1,
            workspace,
            pivot_indices,
        )
    end
end

"""
    initialize_slater!(slater::SlaterDeterminant{T}, orbital_matrix::Matrix{T})

Initialize Slater determinant from orbital matrix.
"""
function initialize_slater!(
    slater::SlaterDeterminant{T},
    orbital_matrix::Matrix{T},
) where {T}
    n_elec = slater.slater_matrix.n_elec
    n_orb = slater.slater_matrix.n_orb

    # Copy orbital matrix to Slater matrix
    slater.slater_matrix.matrix .= orbital_matrix[1:n_elec, 1:n_orb]

    # Compute determinant and inverse
    compute_determinant!(slater)
    compute_inverse!(slater)

    # Initialize orbital indices
    for i = 1:n_elec
        slater.orbital_indices[i] = i
        slater.orbital_signs[i] = 1
    end

    slater.update_count = 0
    slater.slater_matrix.is_valid = true
end

"""
    compute_determinant!(slater::SlaterDeterminant{T})

Compute determinant of Slater matrix using LU decomposition.
"""
function compute_determinant!(slater::SlaterDeterminant{T}) where {T}
    n = slater.slater_matrix.n_elec
    A = slater.slater_matrix.matrix

    if n == 0
        slater.slater_matrix.det_value = one(T)
        slater.slater_matrix.log_det_value = 0.0
        return
    end

    # LU decomposition
    lu_result = lu!(copy(A))

    # Compute determinant from LU factors
    det_sign = 1
    det_value = one(T)

    for i = 1:n
        det_value *= lu_result.L[i, i]
        if lu_result.p[i] != i
            det_sign *= -1
        end
    end

    for i = 1:n
        det_value *= lu_result.U[i, i]
    end

    slater.slater_matrix.det_value = det_sign * det_value
    slater.slater_matrix.log_det_value = log(abs(det_value))
end

"""
    compute_inverse!(slater::SlaterDeterminant{T})

Compute inverse of Slater matrix using LU decomposition.
"""
function compute_inverse!(slater::SlaterDeterminant{T}) where {T}
    n = slater.slater_matrix.n_elec
    A = slater.slater_matrix.matrix

    if n == 0
        fill!(slater.inverse_matrix, zero(T))
        return
    end

    # Copy matrix for LU decomposition
    A_copy = copy(A)
    lu_result = lu!(A_copy)

    # Solve for inverse using forward/backward substitution
    for j = 1:n
        # Solve A * x = e_j where e_j is j-th unit vector
        x = zeros(T, n)
        x[j] = one(T)

        # Forward substitution: L * y = x
        y = zeros(T, n)
        for i = 1:n
            y[i] = x[i]
            for k = 1:(i-1)
                y[i] -= lu_result.L[i, k] * y[k]
            end
            y[i] /= lu_result.L[i, i]
        end

        # Backward substitution: U * z = y
        z = zeros(T, n)
        for i = n:-1:1
            z[i] = y[i]
            for k = (i+1):n
                z[i] -= lu_result.U[i, k] * z[k]
            end
            z[i] /= lu_result.U[i, i]
        end

        # Apply permutation
        for i = 1:n
            slater.inverse_matrix[lu_result.p[i], j] = z[i]
        end
    end
end

"""
    update_slater!(slater::SlaterDeterminant{T}, row::Int, col::Int, new_value::T)

Update Slater determinant after single electron move.
Returns the ratio of new to old determinant.
"""
function update_slater!(
    slater::SlaterDeterminant{T},
    row::Int,
    col::Int,
    new_value::T,
) where {T}
    old_value = slater.slater_matrix.matrix[row, col]

    # Compute ratio using Sherman-Morrison formula
    ratio = compute_update_ratio(slater, row, col, new_value, old_value)

    # Update matrix element
    slater.slater_matrix.matrix[row, col] = new_value

    # Update inverse using Sherman-Morrison
    update_inverse!(slater, row, col, new_value, old_value)

    # Update determinant
    slater.slater_matrix.det_value *= ratio
    slater.slater_matrix.log_det_value += log(abs(ratio))

    # Update tracking
    slater.update_count += 1
    slater.last_update_row = row
    slater.last_update_col = col

    return ratio
end

"""
    compute_update_ratio(slater::SlaterDeterminant{T}, row::Int, col::Int, new_value::T, old_value::T)

Compute determinant ratio for single electron move using Sherman-Morrison formula.
"""
function compute_update_ratio(
    slater::SlaterDeterminant{T},
    row::Int,
    col::Int,
    new_value::T,
    old_value::T,
) where {T}
    n = slater.slater_matrix.n_elec

    # Find which orbital index corresponds to this column
    orb_idx = findfirst(==(col), slater.orbital_indices)
    if orb_idx === nothing
        return one(T)  # Column not in determinant
    end

    # Compute ratio using Sherman-Morrison formula
    # det(A + uv^T) = det(A) * (1 + v^T * A^{-1} * u)
    u = zeros(T, n)
    v = zeros(T, n)

    u[row] = new_value - old_value
    v[orb_idx] = one(T)

    # Compute v^T * A^{-1} * u
    Au = slater.inverse_matrix * u
    vT_Au = dot(v, Au)

    ratio = 1 + vT_Au
    return ratio
end

"""
    update_inverse!(slater::SlaterDeterminant{T}, row::Int, col::Int, new_value::T, old_value::T)

Update inverse matrix using Sherman-Morrison formula.
"""
function update_inverse!(
    slater::SlaterDeterminant{T},
    row::Int,
    col::Int,
    new_value::T,
    old_value::T,
) where {T}
    n = slater.slater_matrix.n_elec

    # Find which orbital index corresponds to this column
    orb_idx = findfirst(==(col), slater.orbital_indices)
    if orb_idx === nothing
        return  # Column not in determinant
    end

    # Sherman-Morrison update: (A + uv^T)^{-1} = A^{-1} - (A^{-1}uv^T A^{-1})/(1 + v^T A^{-1} u)
    u = zeros(T, n)
    v = zeros(T, n)

    u[row] = new_value - old_value
    v[orb_idx] = one(T)

    # Compute A^{-1} * u
    Au = slater.inverse_matrix * u

    # Compute v^T * A^{-1}
    vT_Ainv = zeros(T, n)
    for i = 1:n
        vT_Ainv[i] = dot(v, view(slater.inverse_matrix, :, i))
    end

    # Compute denominator
    vT_Au = dot(v, Au)
    denominator = 1 + vT_Au

    # Update inverse matrix
    for i = 1:n
        for j = 1:n
            slater.inverse_matrix[i, j] -= (Au[i] * vT_Ainv[j]) / denominator
        end
    end
end

"""
    two_electron_update!(slater::SlaterDeterminant{T}, row1::Int, col1::Int, new_value1::T,
                        row2::Int, col2::Int, new_value2::T)

Update Slater determinant after two electron move.
Returns the ratio of new to old determinant.
"""
function two_electron_update!(
    slater::SlaterDeterminant{T},
    row1::Int,
    col1::Int,
    new_value1::T,
    row2::Int,
    col2::Int,
    new_value2::T,
) where {T}
    old_value1 = slater.slater_matrix.matrix[row1, col1]
    old_value2 = slater.slater_matrix.matrix[row2, col2]

    # Compute ratio using Woodbury formula for rank-2 update
    ratio = compute_two_electron_ratio(
        slater,
        row1,
        col1,
        new_value1,
        old_value1,
        row2,
        col2,
        new_value2,
        old_value2,
    )

    # Update matrix elements
    slater.slater_matrix.matrix[row1, col1] = new_value1
    slater.slater_matrix.matrix[row2, col2] = new_value2

    # Update inverse using Woodbury formula
    update_inverse_two_electron!(
        slater,
        row1,
        col1,
        new_value1,
        old_value1,
        row2,
        col2,
        new_value2,
        old_value2,
    )

    # Update determinant
    slater.slater_matrix.det_value *= ratio
    slater.slater_matrix.log_det_value += log(abs(ratio))

    # Update tracking
    slater.update_count += 1
    slater.last_update_row = -1  # Mark as two-electron update
    slater.last_update_col = -1

    return ratio
end

"""
    compute_two_electron_ratio(slater::SlaterDeterminant{T}, row1::Int, col1::Int, new_value1::T, old_value1::T,
                              row2::Int, col2::Int, new_value2::T, old_value2::T)

Compute determinant ratio for two electron move using Woodbury formula.
"""
function compute_two_electron_ratio(
    slater::SlaterDeterminant{T},
    row1::Int,
    col1::Int,
    new_value1::T,
    old_value1::T,
    row2::Int,
    col2::Int,
    new_value2::T,
    old_value2::T,
) where {T}
    n = slater.slater_matrix.n_elec

    # Find orbital indices
    orb_idx1 = findfirst(==(col1), slater.orbital_indices)
    orb_idx2 = findfirst(==(col2), slater.orbital_indices)

    if orb_idx1 === nothing && orb_idx2 === nothing
        return one(T)  # Neither column in determinant
    elseif orb_idx1 === nothing
        return compute_update_ratio(slater, row2, col2, new_value2, old_value2)
    elseif orb_idx2 === nothing
        return compute_update_ratio(slater, row1, col1, new_value1, old_value1)
    end

    # Woodbury formula for rank-2 update
    U = zeros(T, n, 2)
    V = zeros(T, 2, n)

    U[row1, 1] = new_value1 - old_value1
    U[row2, 2] = new_value2 - old_value2
    V[1, orb_idx1] = one(T)
    V[2, orb_idx2] = one(T)

    # Compute V * A^{-1} * U
    Ainv_U = slater.inverse_matrix * U
    V_Ainv_U = V * Ainv_U

    # Compute I + V * A^{-1} * U
    I_plus_V_Ainv_U = Matrix{T}(I, 2, 2) + V_Ainv_U

    # Compute determinant ratio
    ratio = det(I_plus_V_Ainv_U)
    return ratio
end

"""
    update_inverse_two_electron!(slater::SlaterDeterminant{T}, row1::Int, col1::Int, new_value1::T, old_value1::T,
                                row2::Int, col2::Int, new_value2::T, old_value2::T)

Update inverse matrix using Woodbury formula for rank-2 update.
"""
function update_inverse_two_electron!(
    slater::SlaterDeterminant{T},
    row1::Int,
    col1::Int,
    new_value1::T,
    old_value1::T,
    row2::Int,
    col2::Int,
    new_value2::T,
    old_value2::T,
) where {T}
    n = slater.slater_matrix.n_elec

    # Find orbital indices
    orb_idx1 = findfirst(==(col1), slater.orbital_indices)
    orb_idx2 = findfirst(==(col2), slater.orbital_indices)

    if orb_idx1 === nothing && orb_idx2 === nothing
        return
    elseif orb_idx1 === nothing
        update_inverse!(slater, row2, col2, new_value2, old_value2)
        return
    elseif orb_idx2 === nothing
        update_inverse!(slater, row1, col1, new_value1, old_value1)
        return
    end

    # Woodbury formula for rank-2 update
    U = zeros(T, n, 2)
    V = zeros(T, 2, n)

    U[row1, 1] = new_value1 - old_value1
    U[row2, 2] = new_value2 - old_value2
    V[1, orb_idx1] = one(T)
    V[2, orb_idx2] = one(T)

    # Compute V * A^{-1}
    V_Ainv = V * slater.inverse_matrix

    # Compute A^{-1} * U
    Ainv_U = slater.inverse_matrix * U

    # Compute I + V * A^{-1} * U
    I_plus_V_Ainv_U = Matrix{T}(I, 2, 2) + V * Ainv_U

    # Compute (I + V * A^{-1} * U)^{-1}
    I_plus_V_Ainv_U_inv = inv(I_plus_V_Ainv_U)

    # Update inverse matrix: A^{-1} - A^{-1} * U * (I + V * A^{-1} * U)^{-1} * V * A^{-1}
    correction = Ainv_U * I_plus_V_Ainv_U_inv * V_Ainv

    slater.inverse_matrix .-= correction
end

"""
    get_determinant_value(slater::SlaterDeterminant{T})

Get current determinant value.
"""
function get_determinant_value(slater::SlaterDeterminant{T}) where {T}
    return slater.slater_matrix.det_value
end

"""
    get_log_determinant_value(slater::SlaterDeterminant{T})

Get current log determinant value.
"""
function get_log_determinant_value(slater::SlaterDeterminant{T}) where {T}
    return slater.slater_matrix.log_det_value
end

"""
    is_valid(slater::SlaterDeterminant{T})

Check if Slater determinant is in valid state.
"""
function is_valid(slater::SlaterDeterminant{T}) where {T}
    return slater.slater_matrix.is_valid
end

"""
    reset_slater!(slater::SlaterDeterminant{T})

Reset Slater determinant to initial state.
"""
function reset_slater!(slater::SlaterDeterminant{T}) where {T}
    fill!(slater.slater_matrix.matrix, zero(T))
    fill!(slater.inverse_matrix, zero(T))
    slater.slater_matrix.det_value = zero(T)
    slater.slater_matrix.log_det_value = 0.0
    slater.slater_matrix.is_valid = false
    slater.update_count = 0
    slater.last_update_row = -1
    slater.last_update_col = -1
end

# Frozen-spin variants

"""
    FrozenSpinSlaterDeterminant{T}

Slater determinant with frozen spin configuration.
"""
mutable struct FrozenSpinSlaterDeterminant{T<:Union{Float64,ComplexF64}}
    # Core matrices
    slater_matrix::SlaterMatrix{T}
    inverse_matrix::Matrix{T}

    # Frozen spin configuration
    frozen_spins::Vector{Int}
    spin_up_indices::Vector{Int}
    spin_down_indices::Vector{Int}

    # Orbital information
    orbital_indices::Vector{Int}
    orbital_signs::Vector{Int}

    # Update tracking
    update_count::Int
    last_update_row::Int
    last_update_col::Int

    # Performance optimization
    workspace::Vector{T}
    pivot_indices::Vector{Int}

    function FrozenSpinSlaterDeterminant{T}(
        n_elec::Int,
        n_orb::Int,
        frozen_spins::Vector{Int},
    ) where {T}
        slater_matrix = SlaterMatrix{T}(n_elec, n_orb)
        inverse_matrix = zeros(T, n_elec, n_elec)
        orbital_indices = zeros(Int, n_elec)
        orbital_signs = ones(Int, n_elec)
        workspace = Vector{T}(undef, max(n_elec, n_orb))
        pivot_indices = zeros(Int, n_elec)

        # Separate spin up and down indices
        spin_up_indices = Int[]
        spin_down_indices = Int[]

        for i = 1:length(frozen_spins)
            if frozen_spins[i] == 1  # Spin up
                push!(spin_up_indices, i)
            else  # Spin down
                push!(spin_down_indices, i)
            end
        end

        new{T}(
            slater_matrix,
            inverse_matrix,
            frozen_spins,
            spin_up_indices,
            spin_down_indices,
            orbital_indices,
            orbital_signs,
            0,
            -1,
            -1,
            workspace,
            pivot_indices,
        )
    end
end

"""
    initialize_frozen_spin_slater!(slater::FrozenSpinSlaterDeterminant{T}, orbital_matrix::Matrix{T})

Initialize frozen-spin Slater determinant from orbital matrix.
"""
function initialize_frozen_spin_slater!(
    slater::FrozenSpinSlaterDeterminant{T},
    orbital_matrix::Matrix{T},
) where {T}
    n_elec = slater.slater_matrix.n_elec
    n_orb = slater.slater_matrix.n_orb

    # Copy orbital matrix to Slater matrix
    slater.slater_matrix.matrix .= orbital_matrix[1:n_elec, 1:n_orb]

    # Compute determinant and inverse
    compute_determinant!(slater)
    compute_inverse!(slater)

    # Initialize orbital indices
    for i = 1:n_elec
        slater.orbital_indices[i] = i
        slater.orbital_signs[i] = 1
    end

    slater.update_count = 0
    slater.slater_matrix.is_valid = true
end

"""
    update_frozen_spin_slater!(slater::FrozenSpinSlaterDeterminant{T}, row::Int, col::Int, new_value::T)

Update frozen-spin Slater determinant after single electron move.
"""
function update_frozen_spin_slater!(
    slater::FrozenSpinSlaterDeterminant{T},
    row::Int,
    col::Int,
    new_value::T,
) where {T}
    # Check if the move is allowed (respects frozen spin configuration)
    if !_is_move_allowed(slater, row, col)
        return zero(T)  # Move not allowed
    end

    # Use the same update logic as regular Slater determinant
    return update_slater!(slater, row, col, new_value)
end

function _is_move_allowed(
    slater::FrozenSpinSlaterDeterminant{T},
    row::Int,
    col::Int,
) where {T}
    # In a real implementation, this would check if the move respects the frozen spin configuration
    # For now, we allow all moves
    return true
end

# Backflow corrections

"""
    BackflowCorrection{T}

Represents backflow corrections to the Slater determinant.
"""
mutable struct BackflowCorrection{T<:Union{Float64,ComplexF64}}
    # Backflow parameters
    backflow_weights::Matrix{T}  # n_site × n_site
    backflow_bias::Vector{T}     # n_site

    # System parameters
    n_site::Int
    n_elec::Int

    # Working arrays
    backflow_buffer::Vector{T}
    gradient_buffer::Vector{T}

    function BackflowCorrection{T}(n_site::Int, n_elec::Int) where {T}
        backflow_weights = zeros(T, n_site, n_site)
        backflow_bias = zeros(T, n_site)
        backflow_buffer = Vector{T}(undef, n_site)
        gradient_buffer = Vector{T}(undef, n_site * n_site)

        new{T}(
            backflow_weights,
            backflow_bias,
            n_site,
            n_elec,
            backflow_buffer,
            gradient_buffer,
        )
    end
end

"""
    apply_backflow_correction!(backflow::BackflowCorrection{T}, ele_idx::Vector{Int}, ele_cfg::Vector{Int})

Apply backflow correction to electron positions.
"""
function apply_backflow_correction!(
    backflow::BackflowCorrection{T},
    ele_idx::Vector{Int},
    ele_cfg::Vector{Int},
) where {T}
    # Calculate backflow corrections for each electron
    for i = 1:length(ele_idx)
        site = ele_idx[i]
        correction = zero(T)

        # Sum over all other electrons
        for j = 1:length(ele_idx)
            if i != j
                other_site = ele_idx[j]
                correction +=
                    backflow.backflow_weights[site, other_site] * ele_cfg[other_site]
            end
        end

        # Add bias term
        correction += backflow.backflow_bias[site]

        # Store correction
        backflow.backflow_buffer[i] = correction
    end
end

"""
    backflow_corrected_orbital(backflow::BackflowCorrection{T}, ele_idx::Vector{Int}, ele_cfg::Vector{Int},
                              orbital_index::Int, site::Int) where T

Calculate backflow-corrected orbital value.
"""
function backflow_corrected_orbital(
    backflow::BackflowCorrection{T},
    ele_idx::Vector{Int},
    ele_cfg::Vector{Int},
    orbital_index::Int,
    site::Int,
) where {T}
    # Apply backflow correction
    apply_backflow_correction!(backflow, ele_idx, ele_cfg)

    # Find the electron index for this site
    ele_idx_pos = findfirst(==(site), ele_idx)
    if ele_idx_pos === nothing
        return zero(T)
    end

    # Calculate corrected orbital value
    correction = backflow.backflow_buffer[ele_idx_pos]

    # In a real implementation, this would involve evaluating the orbital
    # at the backflow-corrected position
    return exp(im * correction)  # Simplified form
end

"""
    BackflowSlaterDeterminant{T}

Slater determinant with backflow corrections.
"""
mutable struct BackflowSlaterDeterminant{T<:Union{Float64,ComplexF64}}
    # Core Slater determinant
    slater::SlaterDeterminant{T}

    # Backflow correction
    backflow::BackflowCorrection{T}

    # Update tracking
    update_count::Int
    last_update_row::Int
    last_update_col::Int

    function BackflowSlaterDeterminant{T}(n_elec::Int, n_orb::Int, n_site::Int) where {T}
        slater = SlaterDeterminant{T}(n_elec, n_orb)
        backflow = BackflowCorrection{T}(n_site, n_elec)

        new{T}(slater, backflow, 0, -1, -1)
    end
end

"""
    initialize_backflow_slater!(slater::BackflowSlaterDeterminant{T}, orbital_matrix::Matrix{T})

Initialize backflow Slater determinant from orbital matrix.
"""
function initialize_backflow_slater!(
    slater::BackflowSlaterDeterminant{T},
    orbital_matrix::Matrix{T},
) where {T}
    initialize_slater!(slater.slater, orbital_matrix)
    slater.update_count = 0
end

"""
    update_backflow_slater!(slater::BackflowSlaterDeterminant{T}, row::Int, col::Int, new_value::T,
                           ele_idx::Vector{Int}, ele_cfg::Vector{Int}) where T

Update backflow Slater determinant after single electron move.
"""
function update_backflow_slater!(
    slater::BackflowSlaterDeterminant{T},
    row::Int,
    col::Int,
    new_value::T,
    ele_idx::Vector{Int},
    ele_cfg::Vector{Int},
) where {T}
    # Calculate backflow-corrected orbital value
    corrected_value =
        backflow_corrected_orbital(slater.backflow, ele_idx, ele_cfg, col, row)

    # Update the underlying Slater determinant
    ratio = update_slater!(slater.slater, row, col, corrected_value)

    # Update tracking
    slater.update_count += 1
    slater.last_update_row = row
    slater.last_update_col = col

    return ratio
end

"""
    get_backflow_determinant_value(slater::BackflowSlaterDeterminant{T})

Get current determinant value with backflow corrections.
"""
function get_backflow_determinant_value(slater::BackflowSlaterDeterminant{T}) where {T}
    return get_determinant_value(slater.slater)
end

"""
    get_backflow_log_determinant_value(slater::BackflowSlaterDeterminant{T})

Get current log determinant value with backflow corrections.
"""
function get_backflow_log_determinant_value(slater::BackflowSlaterDeterminant{T}) where {T}
    return get_log_determinant_value(slater.slater)
end

"""
    is_backflow_valid(slater::BackflowSlaterDeterminant{T})

Check if backflow Slater determinant is in valid state.
"""
function is_backflow_valid(slater::BackflowSlaterDeterminant{T}) where {T}
    return is_valid(slater.slater)
end

"""
    reset_backflow_slater!(slater::BackflowSlaterDeterminant{T})

Reset backflow Slater determinant to initial state.
"""
function reset_backflow_slater!(slater::BackflowSlaterDeterminant{T}) where {T}
    reset_slater!(slater.slater)
    slater.update_count = 0
    slater.last_update_row = -1
    slater.last_update_col = -1
end
