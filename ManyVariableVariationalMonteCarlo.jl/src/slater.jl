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
mutable struct SlaterMatrix{T <: Union{Float64, ComplexF64}}
    matrix::Matrix{T}
    n_elec::Int
    n_orb::Int
    det_value::T
    log_det_value::Float64
    is_valid::Bool
end

function SlaterMatrix{T}(n_elec::Int, n_orb::Int) where T <: Union{Float64, ComplexF64}
    matrix = zeros(T, n_elec, n_orb)
    return SlaterMatrix{T}(matrix, n_elec, n_orb, zero(T), 0.0, false)
end

"""
    SlaterDeterminant{T}

Main Slater determinant structure containing all necessary components.
"""
mutable struct SlaterDeterminant{T <: Union{Float64, ComplexF64}}
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

    function SlaterDeterminant{T}(n_elec::Int, n_orb::Int) where T <: Union{Float64, ComplexF64}
        slater_matrix = SlaterMatrix{T}(n_elec, n_orb)
        inverse_matrix = zeros(T, n_elec, n_elec)
        orbital_indices = zeros(Int, n_elec)
        orbital_signs = ones(Int, n_elec)
        workspace = Vector{T}(undef, max(n_elec, n_orb))
        pivot_indices = zeros(Int, n_elec)

        new{T}(slater_matrix, inverse_matrix, orbital_indices, orbital_signs,
               0, -1, -1, workspace, pivot_indices)
    end
end

"""
    initialize_slater!(slater::SlaterDeterminant{T}, orbital_matrix::Matrix{T})

Initialize Slater determinant from orbital matrix.
"""
function initialize_slater!(slater::SlaterDeterminant{T}, orbital_matrix::Matrix{T}) where T
    n_elec = slater.slater_matrix.n_elec
    n_orb = slater.slater_matrix.n_orb

    # Copy orbital matrix to Slater matrix
    slater.slater_matrix.matrix .= orbital_matrix[1:n_elec, 1:n_orb]

    # Compute determinant and inverse
    compute_determinant!(slater)
    compute_inverse!(slater)

    # Initialize orbital indices
    for i in 1:n_elec
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
function compute_determinant!(slater::SlaterDeterminant{T}) where T
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

    for i in 1:n
        det_value *= lu_result.L[i, i]
        if lu_result.p[i] != i
            det_sign *= -1
        end
    end

    for i in 1:n
        det_value *= lu_result.U[i, i]
    end

    slater.slater_matrix.det_value = det_sign * det_value
    slater.slater_matrix.log_det_value = log(abs(det_value))
end

"""
    compute_inverse!(slater::SlaterDeterminant{T})

Compute inverse of Slater matrix using LU decomposition.
"""
function compute_inverse!(slater::SlaterDeterminant{T}) where T
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
    for j in 1:n
        # Solve A * x = e_j where e_j is j-th unit vector
        x = zeros(T, n)
        x[j] = one(T)

        # Forward substitution: L * y = x
        y = zeros(T, n)
        for i in 1:n
            y[i] = x[i]
            for k in 1:(i-1)
                y[i] -= lu_result.L[i, k] * y[k]
            end
            y[i] /= lu_result.L[i, i]
        end

        # Backward substitution: U * z = y
        z = zeros(T, n)
        for i in n:-1:1
            z[i] = y[i]
            for k in (i+1):n
                z[i] -= lu_result.U[i, k] * z[k]
            end
            z[i] /= lu_result.U[i, i]
        end

        # Apply permutation
        for i in 1:n
            slater.inverse_matrix[lu_result.p[i], j] = z[i]
        end
    end
end

"""
    update_slater!(slater::SlaterDeterminant{T}, row::Int, col::Int, new_value::T)

Update Slater determinant after single electron move.
Returns the ratio of new to old determinant.
"""
function update_slater!(slater::SlaterDeterminant{T}, row::Int, col::Int, new_value::T) where T
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
function compute_update_ratio(slater::SlaterDeterminant{T}, row::Int, col::Int, new_value::T, old_value::T) where T
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
function update_inverse!(slater::SlaterDeterminant{T}, row::Int, col::Int, new_value::T, old_value::T) where T
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
    for i in 1:n
        vT_Ainv[i] = dot(v, view(slater.inverse_matrix, :, i))
    end

    # Compute denominator
    vT_Au = dot(v, Au)
    denominator = 1 + vT_Au

    # Update inverse matrix
    for i in 1:n
        for j in 1:n
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
function two_electron_update!(slater::SlaterDeterminant{T}, row1::Int, col1::Int, new_value1::T,
                              row2::Int, col2::Int, new_value2::T) where T
    old_value1 = slater.slater_matrix.matrix[row1, col1]
    old_value2 = slater.slater_matrix.matrix[row2, col2]

    # Compute ratio using Woodbury formula for rank-2 update
    ratio = compute_two_electron_ratio(slater, row1, col1, new_value1, old_value1,
                                       row2, col2, new_value2, old_value2)

    # Update matrix elements
    slater.slater_matrix.matrix[row1, col1] = new_value1
    slater.slater_matrix.matrix[row2, col2] = new_value2

    # Update inverse using Woodbury formula
    update_inverse_two_electron!(slater, row1, col1, new_value1, old_value1,
                                 row2, col2, new_value2, old_value2)

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
function compute_two_electron_ratio(slater::SlaterDeterminant{T}, row1::Int, col1::Int, new_value1::T, old_value1::T,
                                   row2::Int, col2::Int, new_value2::T, old_value2::T) where T
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
function update_inverse_two_electron!(slater::SlaterDeterminant{T}, row1::Int, col1::Int, new_value1::T, old_value1::T,
                                     row2::Int, col2::Int, new_value2::T, old_value2::T) where T
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
function get_determinant_value(slater::SlaterDeterminant{T}) where T
    return slater.slater_matrix.det_value
end

"""
    get_log_determinant_value(slater::SlaterDeterminant{T})

Get current log determinant value.
"""
function get_log_determinant_value(slater::SlaterDeterminant{T}) where T
    return slater.slater_matrix.log_det_value
end

"""
    is_valid(slater::SlaterDeterminant{T})

Check if Slater determinant is in valid state.
"""
function is_valid(slater::SlaterDeterminant{T}) where T
    return slater.slater_matrix.is_valid
end

"""
    reset_slater!(slater::SlaterDeterminant{T})

Reset Slater determinant to initial state.
"""
function reset_slater!(slater::SlaterDeterminant{T}) where T
    fill!(slater.slater_matrix.matrix, zero(T))
    fill!(slater.inverse_matrix, zero(T))
    slater.slater_matrix.det_value = zero(T)
    slater.slater_matrix.log_det_value = 0.0
    slater.slater_matrix.is_valid = false
    slater.update_count = 0
    slater.last_update_row = -1
    slater.last_update_col = -1
end
