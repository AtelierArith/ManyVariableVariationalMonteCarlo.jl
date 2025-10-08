"""
Slater Determinant System for mVMC C Compatibility

Translates the Slater determinant modules (slater.c, slater_fsz.c) to Julia,
maintaining exact compatibility with C numerical methods and Pfaffian calculations.

Ported from slater.c and slater_fsz.c.
"""

using LinearAlgebra
using SparseArrays

# Constants matching C implementation
const EPS = 1.0e-14
const MAX_ITER = 100

"""
    MVMCSlaterDeterminant

Slater determinant state matching the C implementation.
"""
mutable struct MVMCSlaterDeterminant
    # System parameters
    nsite::Int
    ne::Int
    nup::Int
    nsize::Int
    nqp_full::Int

    # Slater matrices
    slater_elm::Union{Array{ComplexF64, 3}, Nothing}
    inv_m::Union{Array{ComplexF64, 3}, Nothing}
    pf_m::Union{Vector{ComplexF64}, Nothing}

    # Real versions
    slater_elm_real::Union{Array{Float64, 3}, Nothing}
    inv_m_real::Union{Array{Float64, 3}, Nothing}
    pf_m_real::Union{Vector{Float64}, Nothing}

    # Backflow versions
    slater_elm_bf::Union{Array{ComplexF64, 3}, Nothing}
    slater_elm_bf_real::Union{Array{Float64, 3}, Nothing}

    # Electron configuration
    ele_idx::Union{Vector{Int}, Nothing}
    ele_cfg::Union{Vector{Int}, Nothing}
    ele_num::Union{Vector{Int}, Nothing}
    ele_spn::Union{Vector{Int}, Nothing}

    # Temporary arrays
    tmp_ele_idx::Union{Vector{Int}, Nothing}
    tmp_ele_cfg::Union{Vector{Int}, Nothing}
    tmp_ele_num::Union{Vector{Int}, Nothing}
    tmp_ele_spn::Union{Vector{Int}, Nothing}

    # Wavefunction values
    log_sq_pf_full_slater::Union{Vector{Float64}, Nothing}
    smp_slt_elm_bf_real::Union{Vector{Float64}, Nothing}

    # Flags
    use_real::Bool
    use_backflow::Bool
    use_fsz::Bool

    function MVMCSlaterDeterminant()
        new(
            0,      # nsite
            0,      # ne
            0,      # nup
            0,      # nsize
            0,      # nqp_full
            nothing, # slater_elm
            nothing, # inv_m
            nothing, # pf_m
            nothing, # slater_elm_real
            nothing, # inv_m_real
            nothing, # pf_m_real
            nothing, # slater_elm_bf
            nothing, # slater_elm_bf_real
            nothing, # ele_idx
            nothing, # ele_cfg
            nothing, # ele_num
            nothing, # ele_spn
            nothing, # tmp_ele_idx
            nothing, # tmp_ele_cfg
            nothing, # tmp_ele_num
            nothing, # tmp_ele_spn
            nothing, # log_sq_pf_full_slater
            nothing, # smp_slt_elm_bf_real
            false,  # use_real
            false,  # use_backflow
            false   # use_fsz
        )
    end
end

"""
    initialize_slater!(slater::MVMCSlaterDeterminant, state::MVMCGlobalState)

Initialize Slater determinant system.
Matches C function initialize_slater().

C実装参考: slater.c 1行目から1506行目まで
"""
function initialize_slater!(slater::MVMCSlaterDeterminant, state::MVMCGlobalState)
    # Set system parameters
    slater.nsite = state.nsite
    slater.ne = state.ne
    slater.nup = state.nup
    slater.nsize = state.nsize
    slater.nqp_full = state.nqp_full

    # Set flags
    slater.use_real = state.all_complex_flag == 0
    slater.use_backflow = state.n_back_flow_idx > 0
    slater.use_fsz = state.two_sz != 0

    # Allocate arrays
    allocate_slater_arrays!(slater, state)

    # Initialize electron configuration
    initialize_electron_configuration!(slater, state)

    # Initialize Slater matrices
    initialize_slater_matrices!(slater, state)
end

"""
    allocate_slater_arrays!(slater::MVMCSlaterDeterminant, state::MVMCGlobalState)

Allocate Slater determinant arrays.
"""
function allocate_slater_arrays!(slater::MVMCSlaterDeterminant, state::MVMCGlobalState)
    nsite = slater.nsite
    ne = slater.ne
    nsize = slater.nsize
    nqp_full = slater.nqp_full

    # Electron configuration arrays
    slater.ele_idx = Vector{Int}(undef, nsize)
    slater.ele_cfg = Vector{Int}(undef, nsite * 2)
    slater.ele_num = Vector{Int}(undef, nsite * 2)
    slater.ele_spn = Vector{Int}(undef, nsize)

    # Temporary arrays
    slater.tmp_ele_idx = Vector{Int}(undef, nsize)
    slater.tmp_ele_cfg = Vector{Int}(undef, nsite * 2)
    slater.tmp_ele_num = Vector{Int}(undef, nsite * 2)
    slater.tmp_ele_spn = Vector{Int}(undef, nsize)

    # Wavefunction arrays
    slater.log_sq_pf_full_slater = Vector{Float64}(undef, state.nvmc_sample)
    slater.smp_slt_elm_bf_real = Vector{Float64}(undef, state.nvmc_sample)

    # Slater matrices
    if nqp_full > 0
        slater.slater_elm = Array{ComplexF64, 3}(undef, nqp_full, nsite * 2, nsite * 2)
        slater.inv_m = Array{ComplexF64, 3}(undef, nqp_full, ne, ne)
        slater.pf_m = Vector{ComplexF64}(undef, nqp_full)

        # Real versions
        if slater.use_real
            slater.slater_elm_real = Array{Float64, 3}(undef, nqp_full, nsite * 2, nsite * 2)
            slater.inv_m_real = Array{Float64, 3}(undef, nqp_full, ne, ne)
            slater.pf_m_real = Vector{Float64}(undef, nqp_full)
        end

        # Backflow versions
        if slater.use_backflow
            slater.slater_elm_bf = Array{ComplexF64, 3}(undef, nqp_full, nsite * 2, nsite * 2)
            if slater.use_real
                slater.slater_elm_bf_real = Array{Float64, 3}(undef, nqp_full, nsite * 2, nsite * 2)
            end
        end
    end
end

"""
    initialize_electron_configuration!(slater::MVMCSlaterDeterminant, state::MVMCGlobalState)

Initialize electron configuration.
Matches C function initialize_electron_configuration().
"""
function initialize_electron_configuration!(slater::MVMCSlaterDeterminant, state::MVMCGlobalState)
    nsite = slater.nsite
    ne = slater.ne
    nup = slater.nup
    ndown = ne - nup

    # Initialize electron indices
    idx = 1
    for i in 1:nup
        slater.ele_idx[idx] = i
        slater.ele_spn[idx] = 1  # up spin
        idx += 1
    end
    for i in 1:ndown
        slater.ele_idx[idx] = i
        slater.ele_spn[idx] = -1  # down spin
        idx += 1
    end

    # Initialize electron configuration
    fill!(slater.ele_cfg, 0)
    fill!(slater.ele_num, 0)

    for i in 1:ne
        site = slater.ele_idx[i]
        spin = slater.ele_spn[i]
        if spin > 0
            slater.ele_cfg[site] = 1
            slater.ele_num[site] = 1
        else
            slater.ele_cfg[site + nsite] = 1
            slater.ele_num[site + nsite] = 1
        end
    end
end

"""
    initialize_slater_matrices!(slater::MVMCSlaterDeterminant, state::MVMCGlobalState)

Initialize Slater matrices.
Matches C function initialize_slater_matrices().
"""
function initialize_slater_matrices!(slater::MVMCSlaterDeterminant, state::MVMCGlobalState)
    nsite = slater.nsite
    ne = slater.ne
    nqp_full = slater.nqp_full

    if nqp_full == 0
        return
    end

    # Initialize Slater matrices for each quantum projection
    for qp in 1:nqp_full
        # Initialize Slater matrix elements
        if slater.use_real
            initialize_slater_matrix_real!(slater, state, qp)
        else
            initialize_slater_matrix_complex!(slater, state, qp)
        end

        # Calculate Pfaffian
        calculate_pfaffian!(slater, qp)

        # Calculate inverse matrix
        calculate_inverse_matrix!(slater, qp)
    end
end

"""
    initialize_slater_matrix_complex!(slater::MVMCSlaterDeterminant, state::MVMCGlobalState, qp::Int)

Initialize complex Slater matrix.
"""
function initialize_slater_matrix_complex!(slater::MVMCSlaterDeterminant, state::MVMCGlobalState, qp::Int)
    nsite = slater.nsite
    ne = slater.ne

    # Initialize Slater matrix elements
    for i in 1:ne
        for j in 1:ne
            site_i = slater.ele_idx[i]
            site_j = slater.ele_idx[j]
            spin_i = slater.ele_spn[i]
            spin_j = slater.ele_spn[j]

            # Calculate matrix element
            if spin_i > 0 && spin_j > 0
                # Up-up
                slater.slater_elm[qp, i, j] = state.slater_elm[qp, site_i, site_j]
            elseif spin_i < 0 && spin_j < 0
                # Down-down
                slater.slater_elm[qp, i, j] = state.slater_elm[qp, site_i + nsite, site_j + nsite]
            else
                # Up-down or down-up (should be zero for Sz conservation)
                slater.slater_elm[qp, i, j] = ComplexF64(0.0, 0.0)
            end
        end
    end
end

"""
    initialize_slater_matrix_real!(slater::MVMCSlaterDeterminant, state::MVMCGlobalState, qp::Int)

Initialize real Slater matrix.
"""
function initialize_slater_matrix_real!(slater::MVMCSlaterDeterminant, state::MVMCGlobalState, qp::Int)
    nsite = slater.nsite
    ne = slater.ne

    # Initialize Slater matrix elements
    for i in 1:ne
        for j in 1:ne
            site_i = slater.ele_idx[i]
            site_j = slater.ele_idx[j]
            spin_i = slater.ele_spn[i]
            spin_j = slater.ele_spn[j]

            # Calculate matrix element
            if spin_i > 0 && spin_j > 0
                # Up-up
                slater.slater_elm_real[qp, i, j] = real(state.slater_elm[qp, site_i, site_j])
            elseif spin_i < 0 && spin_j < 0
                # Down-down
                slater.slater_elm_real[qp, i, j] = real(state.slater_elm[qp, site_i + nsite, site_j + nsite])
            else
                # Up-down or down-up (should be zero for Sz conservation)
                slater.slater_elm_real[qp, i, j] = 0.0
            end
        end
    end
end

"""
    calculate_pfaffian!(slater::MVMCSlaterDeterminant, qp::Int)

Calculate Pfaffian of Slater matrix.
Matches C function calculate_pfaffian().
"""
function calculate_pfaffian!(slater::MVMCSlaterDeterminant, qp::Int)
    ne = slater.ne

    if ne == 0
        slater.pf_m[qp] = ComplexF64(1.0, 0.0)
        if slater.use_real
            slater.pf_m_real[qp] = 1.0
        end
        return
    end

    if slater.use_real
        # Real Pfaffian calculation
        matrix = slater.slater_elm_real[qp, 1:ne, 1:ne]
        pfaffian_val = pfaffian_skew_symmetric(matrix)
        slater.pf_m_real[qp] = pfaffian_val
        slater.pf_m[qp] = ComplexF64(pfaffian_val, 0.0)
    else
        # Complex Pfaffian calculation
        matrix = slater.slater_elm[qp, 1:ne, 1:ne]
        pfaffian_val = pfaffian_skew_symmetric(matrix)
        slater.pf_m[qp] = pfaffian_val
        if slater.use_real
            slater.pf_m_real[qp] = real(pfaffian_val)
        end
    end
end

"""
    calculate_inverse_matrix!(slater::MVMCSlaterDeterminant, qp::Int)

Calculate inverse of Slater matrix.
Matches C function calculate_inverse_matrix().
"""
function calculate_inverse_matrix!(slater::MVMCSlaterDeterminant, qp::Int)
    ne = slater.ne

    if ne == 0
        return
    end

    if slater.use_real
        # Real inverse calculation
        matrix = slater.slater_elm_real[qp, 1:ne, 1:ne]
        inv_matrix = inv(matrix)
        slater.inv_m_real[qp, 1:ne, 1:ne] = inv_matrix
        slater.inv_m[qp, 1:ne, 1:ne] = ComplexF64.(inv_matrix)
    else
        # Complex inverse calculation
        matrix = slater.slater_elm[qp, 1:ne, 1:ne]
        inv_matrix = inv(matrix)
        slater.inv_m[qp, 1:ne, 1:ne] = inv_matrix
        if slater.use_real
            slater.inv_m_real[qp, 1:ne, 1:ne] = real.(inv_matrix)
        end
    end
end

"""
    update_slater_matrix!(slater::MVMCSlaterDeterminant, state::MVMCGlobalState, qp::Int, i::Int, j::Int)

Update Slater matrix after electron move.
Matches C function update_slater_matrix().
"""
function update_slater_matrix!(slater::MVMCSlaterDeterminant, state::MVMCGlobalState, qp::Int, i::Int, j::Int)
    ne = slater.ne

    if i < 1 || i > ne || j < 1 || j > ne
        error("Invalid electron indices: i=$i, j=$j, ne=$ne")
    end

    # Update matrix elements
    if slater.use_real
        update_slater_matrix_real!(slater, state, qp, i, j)
    else
        update_slater_matrix_complex!(slater, state, qp, i, j)
    end

    # Recalculate Pfaffian and inverse
    calculate_pfaffian!(slater, qp)
    calculate_inverse_matrix!(slater, qp)
end

"""
    update_slater_matrix_complex!(slater::MVMCSlaterDeterminant, state::MVMCGlobalState, qp::Int, i::Int, j::Int)

Update complex Slater matrix.
"""
function update_slater_matrix_complex!(slater::MVMCSlaterDeterminant, state::MVMCGlobalState, qp::Int, i::Int, j::Int)
    nsite = slater.nsite

    # Update matrix elements
    for k in 1:slater.ne
        site_i = slater.ele_idx[i]
        site_k = slater.ele_idx[k]
        spin_i = slater.ele_spn[i]
        spin_k = slater.ele_spn[k]

        if spin_i > 0 && spin_k > 0
            # Up-up
            slater.slater_elm[qp, i, k] = state.slater_elm[qp, site_i, site_k]
        elseif spin_i < 0 && spin_k < 0
            # Down-down
            slater.slater_elm[qp, i, k] = state.slater_elm[qp, site_i + nsite, site_k + nsite]
        else
            # Up-down or down-up
            slater.slater_elm[qp, i, k] = ComplexF64(0.0, 0.0)
        end
    end

    for k in 1:slater.ne
        site_j = slater.ele_idx[j]
        site_k = slater.ele_idx[k]
        spin_j = slater.ele_spn[j]
        spin_k = slater.ele_spn[k]

        if spin_j > 0 && spin_k > 0
            # Up-up
            slater.slater_elm[qp, k, j] = state.slater_elm[qp, site_k, site_j]
        elseif spin_j < 0 && spin_k < 0
            # Down-down
            slater.slater_elm[qp, k, j] = state.slater_elm[qp, site_k + nsite, site_j + nsite]
        else
            # Up-down or down-up
            slater.slater_elm[qp, k, j] = ComplexF64(0.0, 0.0)
        end
    end
end

"""
    update_slater_matrix_real!(slater::MVMCSlaterDeterminant, state::MVMCGlobalState, qp::Int, i::Int, j::Int)

Update real Slater matrix.
"""
function update_slater_matrix_real!(slater::MVMCSlaterDeterminant, state::MVMCGlobalState, qp::Int, i::Int, j::Int)
    nsite = slater.nsite

    # Update matrix elements
    for k in 1:slater.ne
        site_i = slater.ele_idx[i]
        site_k = slater.ele_idx[k]
        spin_i = slater.ele_spn[i]
        spin_k = slater.ele_spn[k]

        if spin_i > 0 && spin_k > 0
            # Up-up
            slater.slater_elm_real[qp, i, k] = real(state.slater_elm[qp, site_i, site_k])
        elseif spin_i < 0 && spin_k < 0
            # Down-down
            slater.slater_elm_real[qp, i, k] = real(state.slater_elm[qp, site_i + nsite, site_k + nsite])
        else
            # Up-down or down-up
            slater.slater_elm_real[qp, i, k] = 0.0
        end
    end

    for k in 1:slater.ne
        site_j = slater.ele_idx[j]
        site_k = slater.ele_idx[k]
        spin_j = slater.ele_spn[j]
        spin_k = slater.ele_spn[k]

        if spin_j > 0 && spin_k > 0
            # Up-up
            slater.slater_elm_real[qp, k, j] = real(state.slater_elm[qp, site_k, site_j])
        elseif spin_j < 0 && spin_k < 0
            # Down-down
            slater.slater_elm_real[qp, k, j] = real(state.slater_elm[qp, site_k + nsite, site_j + nsite])
        else
            # Up-down or down-up
            slater.slater_elm_real[qp, k, j] = 0.0
        end
    end
end

"""
    calculate_wavefunction_value!(slater::MVMCSlaterDeterminant, state::MVMCGlobalState, sample::Int)

Calculate wavefunction value for a sample.
Matches C function calculate_wavefunction_value().
"""
function calculate_wavefunction_value!(slater::MVMCSlaterDeterminant, state::MVMCGlobalState, sample::Int)
    nqp_full = slater.nqp_full

    if nqp_full == 0
        return ComplexF64(1.0, 0.0)
    end

    # Calculate weighted sum over quantum projections
    total_value = ComplexF64(0.0, 0.0)

    for qp in 1:nqp_full
        weight = state.qp_full_weight[qp]
        pfaffian_val = slater.pf_m[qp]
        total_value += weight * pfaffian_val
    end

    # Store log of squared absolute value
    log_sq_value = log(abs2(total_value))
    slater.log_sq_pf_full_slater[sample] = log_sq_value

    return total_value
end

"""
    print_slater_summary(slater::MVMCSlaterDeterminant)

Print Slater determinant summary.
"""
function print_slater_summary(slater::MVMCSlaterDeterminant)
    println("=== Slater Determinant Summary ===")
    println("System: Nsite=$(slater.nsite), Ne=$(slater.ne), Nup=$(slater.nup)")
    println("Quantum projections: $(slater.nqp_full)")
    println("Flags: Real=$(slater.use_real), Backflow=$(slater.use_backflow), FSZ=$(slater.use_fsz)")
    println("==================================")
end

# Export functions and types
export MVMCSlaterDeterminant, initialize_slater!, allocate_slater_arrays!,
       initialize_electron_configuration!, initialize_slater_matrices!,
       calculate_pfaffian!, calculate_inverse_matrix!, update_slater_matrix!,
       calculate_wavefunction_value!, print_slater_summary
