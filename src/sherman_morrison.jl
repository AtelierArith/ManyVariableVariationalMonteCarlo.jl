"""
Sherman-Morrison Update for Slater Determinants

Efficient update of inverse matrix when one row changes.
Used for calculating acceptance ratios in VMC sampling.

Based on mVMC C implementation:
- pfupdate.c: Pfaffian/Determinant updates
- vmccal.c: Ratio calculations
"""

using LinearAlgebra

"""
    calculate_ratio_sherman_morrison(
        inv_m::Matrix{T},
        orbital_matrix::Matrix{T},
        electron_idx::Int,
        site_from::Int,
        site_to::Int,
        electron_list_same_spin::Vector{Int}
    ) where T

Calculate acceptance ratio using Sherman-Morrison formula.

When electron `electron_idx` moves from `site_from` to `site_to`,
the Slater determinant ratio is:

    ratio = det(M_new) / det(M_old)

Using Sherman-Morrison:
    ratio = 1 + u^T * M_old^{-1} * v

where u and v are the row/column changes.

# Arguments
- `inv_m`: Inverse of current Slater matrix [n_elec_spin × n_elec_spin]
- `orbital_matrix`: Orbital matrix [n_sites × n_sites]
- `electron_idx`: Index of electron to move (in same-spin list)
- `site_from`: Current site (0-based)
- `site_to`: New site (0-based)
- `electron_list_same_spin`: List of electron sites with same spin (0-based)

# Returns
- `ratio`: det(M_new) / det(M_old)

C実装参考: pfupdate.c 1行目から608行目まで
"""
function calculate_ratio_sherman_morrison(
    inv_m::Matrix{T},
    orbital_matrix::Matrix{T},
    electron_idx::Int,
    site_from::Int,
    site_to::Int,
    electron_list_same_spin::Vector{Int},
) where {T}
    n_elec_spin = length(electron_list_same_spin)

    if n_elec_spin == 0
        return one(T)
    end

    # Calculate ratio using Sherman-Morrison
    # ratio = sum_j inv_m[electron_idx, j] * (phi[site_to, j] - phi[site_from, j])
    #       = sum_j inv_m[electron_idx, j] * delta_phi[j]

    ratio = zero(T)

    for j in 1:n_elec_spin
        site_j = electron_list_same_spin[j]
        # Ensure indices are within bounds
        if site_to+1 > size(orbital_matrix, 1) || site_from+1 > size(orbital_matrix, 1) ||
           site_j+1 > size(orbital_matrix, 2)
            @warn "Index out of bounds: site_to=$site_to, site_from=$site_from, site_j=$site_j, matrix_size=$(size(orbital_matrix))"
            return one(T)
        end
        delta_phi = orbital_matrix[site_to+1, site_j+1] - orbital_matrix[site_from+1, site_j+1]
        ratio += inv_m[electron_idx, j] * delta_phi
    end

    return ratio
end

"""
    update_inverse_sherman_morrison!(
        inv_m::Matrix{T},
        orbital_matrix::Matrix{T},
        electron_idx::Int,
        site_from::Int,
        site_to::Int,
        electron_list_same_spin::Vector{Int},
        ratio::T
    ) where T

Update inverse matrix using Sherman-Morrison formula after accepted move.

The update formula is:
    M_new^{-1} = M_old^{-1} - (M_old^{-1} * u * v^T * M_old^{-1}) / (1 + v^T * M_old^{-1} * u)

where u is the column change and v is the row change.

# Arguments
- `inv_m`: Inverse matrix to update (modified in-place)
- `orbital_matrix`: Orbital matrix
- `electron_idx`: Index of moved electron (in same-spin list)
- `site_from`: Old site (0-based)
- `site_to`: New site (0-based)
- `electron_list_same_spin`: List of electron sites with same spin (0-based)
- `ratio`: Pre-calculated ratio from calculate_ratio_sherman_morrison
"""
function update_inverse_sherman_morrison!(
    inv_m::Matrix{T},
    orbital_matrix::Matrix{T},
    electron_idx::Int,
    site_from::Int,
    site_to::Int,
    electron_list_same_spin::Vector{Int},
    ratio::T,
) where {T}
    n_elec_spin = length(electron_list_same_spin)

    if n_elec_spin == 0 || abs(ratio) < 1e-15
        return nothing
    end

    # Calculate delta_phi = phi_new - phi_old
    delta_phi = Vector{T}(undef, n_elec_spin)
    for j in 1:n_elec_spin
        site_j = electron_list_same_spin[j]
        delta_phi[j] = orbital_matrix[site_to+1, site_j+1] -
                       orbital_matrix[site_from+1, site_j+1]
    end

    # Calculate beta = M_old^{-1} * delta_phi
    beta = inv_m * delta_phi

    # Update: M_new^{-1} = M_old^{-1} - (alpha * beta^T) / ratio
    # where alpha[i] = inv_m[electron_idx, i]
    inv_ratio = one(T) / ratio

    for i in 1:n_elec_spin
        alpha_i = inv_m[electron_idx, i]
        for j in 1:n_elec_spin
            inv_m[i, j] -= alpha_i * beta[j] * inv_ratio
        end
    end

    return nothing
end

"""
    calculate_determinant_ratio(
        inv_m::Matrix{T},
        new_row::Vector{T},
        row_idx::Int
    ) where T

Calculate determinant ratio when one row of matrix changes.

# Arguments
- `inv_m`: Inverse of old matrix
- `new_row`: New row vector
- `row_idx`: Index of row to replace (1-based)

# Returns
- Ratio: det(M_new) / det(M_old)
"""
function calculate_determinant_ratio(
    inv_m::Matrix{T},
    new_row::Vector{T},
    row_idx::Int,
) where {T}
    # ratio = new_row^T * inv_m[:, row_idx]
    ratio = zero(T)
    n = size(inv_m, 1)

    for i in 1:n
        ratio += new_row[i] * inv_m[i, row_idx]
    end

    return ratio
end

"""
    rebuild_slater_matrix_and_inverse!(
        slater::Matrix{T},
        inv_m::Matrix{T},
        orbital_matrix::Matrix{T},
        electron_list::Vector{Int}
    ) where T

Rebuild Slater matrix and its inverse from scratch.
Use when Sherman-Morrison updates accumulate numerical errors.

# Arguments
- `slater`: Slater matrix to rebuild (modified in-place)
- `inv_m`: Inverse matrix to rebuild (modified in-place)
- `orbital_matrix`: Orbital matrix
- `electron_list`: List of electron sites (0-based)

# Returns
- `det_val`: Determinant of Slater matrix
"""
function rebuild_slater_matrix_and_inverse!(
    slater::Matrix{T},
    inv_m::Matrix{T},
    orbital_matrix::Matrix{T},
    electron_list::Vector{Int},
) where {T}
    n_elec = length(electron_list)

    if n_elec == 0
        return one(T)
    end

    # Build Slater matrix
    for i in 1:n_elec
        site_i = electron_list[i]
        for j in 1:n_elec
            site_j = electron_list[j]
            slater[i, j] = orbital_matrix[site_i+1, site_j+1]
        end
    end

    # Calculate determinant and inverse
    det_val = det(slater)

    if abs(det_val) > 1e-15
        inv_m .= inv(slater)
    else
        # Singular matrix, set to identity as fallback
        inv_m .= Matrix{T}(I, n_elec, n_elec)
    end

    return det_val
end

"""
    check_inverse_accuracy(
        slater::Matrix{T},
        inv_m::Matrix{T};
        threshold::Float64 = 1e-10
    ) where T

Check accuracy of inverse matrix.
Returns true if M * M^{-1} is close to identity.

# Arguments
- `slater`: Slater matrix
- `inv_m`: Claimed inverse matrix
- `threshold`: Maximum deviation from identity

# Returns
- `is_accurate`: true if accurate, false otherwise
- `max_error`: Maximum element-wise error
"""
function check_inverse_accuracy(
    slater::Matrix{T},
    inv_m::Matrix{T};
    threshold::Float64 = 1e-10,
) where {T}
    n = size(slater, 1)

    if n == 0
        return (true, 0.0)
    end

    # Calculate M * M^{-1}
    product = slater * inv_m

    # Check against identity
    max_error = 0.0
    for i in 1:n
        for j in 1:n
            expected = (i == j) ? 1.0 : 0.0
            error = abs(real(product[i, j]) - expected)
            if imag(product[i, j]) != 0
                error = max(error, abs(imag(product[i, j])))
            end
            max_error = max(max_error, error)
        end
    end

    is_accurate = max_error < threshold

    return (is_accurate, max_error)
end

"""
    ShermanMorrisonManager{T}

Manages Sherman-Morrison updates and periodic rebuilds.
"""
mutable struct ShermanMorrisonManager{T<:Number}
    # Frequency of rebuilds (0 = never rebuild)
    rebuild_frequency::Int

    # Counter for updates since last rebuild
    update_count::Int

    # Accuracy threshold for triggering rebuild
    accuracy_threshold::Float64

    # Statistics
    n_updates::Int
    n_rebuilds::Int
    n_forced_rebuilds::Int  # Due to accuracy issues

    function ShermanMorrisonManager{T}(;
        rebuild_frequency::Int = 100,
        accuracy_threshold::Float64 = 1e-8,
    ) where {T}
        new{T}(rebuild_frequency, 0, accuracy_threshold, 0, 0, 0)
    end
end

"""
    should_rebuild(manager::ShermanMorrisonManager) -> Bool

Check if it's time to rebuild matrices.
"""
function should_rebuild(manager::ShermanMorrisonManager)
    if manager.rebuild_frequency <= 0
        return false
    end

    return manager.update_count >= manager.rebuild_frequency
end

"""
    record_update!(manager::ShermanMorrisonManager)

Record that an update was performed.
"""
function record_update!(manager::ShermanMorrisonManager)
    manager.update_count += 1
    manager.n_updates += 1
    return nothing
end

"""
    record_rebuild!(manager::ShermanMorrisonManager, forced::Bool = false)

Record that a rebuild was performed.
"""
function record_rebuild!(manager::ShermanMorrisonManager, forced::Bool = false)
    manager.update_count = 0
    manager.n_rebuilds += 1
    if forced
        manager.n_forced_rebuilds += 1
    end
    return nothing
end

"""
    print_statistics(manager::ShermanMorrisonManager)

Print Sherman-Morrison update statistics.
"""
function print_statistics(manager::ShermanMorrisonManager)
    println("Sherman-Morrison Update Statistics:")
    println("  Total updates: $(manager.n_updates)")
    println("  Total rebuilds: $(manager.n_rebuilds)")
    println("  Forced rebuilds: $(manager.n_forced_rebuilds)")
    if manager.n_rebuilds > 0
        avg_updates = manager.n_updates / manager.n_rebuilds
        println("  Average updates per rebuild: $(round(avg_updates, digits=1))")
    end
end
