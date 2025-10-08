# FSZ (Fixed Spin Configuration) specialized modules
# Based on mVMC C implementation: vmcmake_fsz.c, vmccal_fsz.c, calham_fsz.c, etc.

"""
    FSZConfiguration

Configuration for Fixed Spin Zone (FSZ) calculations.
In FSZ mode, the spin configuration is fixed and only spatial degrees of freedom are sampled.
"""
struct FSZConfiguration{T<:Number}
    # Fixed spin configuration
    spin_config::Vector{Int}  # +1 for up, -1 for down, 0 for empty
    n_up::Int                 # Number of up spins
    n_down::Int              # Number of down spins
    n_sites::Int             # Total number of sites

    # Site indices by spin
    up_sites::Vector{Int}    # Sites with up spins
    down_sites::Vector{Int}  # Sites with down spins
    empty_sites::Vector{Int} # Empty sites

    # Projection counters
    proj_count::Vector{Int}  # Projection count per site

    # FSZ-specific flags
    use_real::Bool          # Use real arithmetic if possible
    optimize_fsz::Bool      # Use FSZ-specific optimizations
end

"""
    create_fsz_configuration(spin_config::Vector{Int}; use_real=false, optimize_fsz=true)

Create FSZ configuration from a given spin configuration.

C実装参考: vmcmake_fsz.c 1行目から855行目まで
"""
function create_fsz_configuration(
    spin_config::Vector{Int};
    use_real = false,
    optimize_fsz = true,
)
    n_sites = length(spin_config)

    up_sites = findall(x -> x == 1, spin_config)
    down_sites = findall(x -> x == -1, spin_config)
    empty_sites = findall(x -> x == 0, spin_config)

    n_up = length(up_sites)
    n_down = length(down_sites)

    proj_count = zeros(Int, n_sites)
    for (i, spin) in enumerate(spin_config)
        if spin != 0
            proj_count[i] = 1
        end
    end

    T = use_real ? Float64 : ComplexF64

    return FSZConfiguration{T}(
        spin_config,
        n_up,
        n_down,
        n_sites,
        up_sites,
        down_sites,
        empty_sites,
        proj_count,
        use_real,
        optimize_fsz,
    )
end

"""
    FSZState{T}

VMC state specialized for fixed spin configurations.
Similar to VMCState but optimized for FSZ calculations.
"""
mutable struct FSZState{T<:Number}
    # Electron configurations (spatial only, spin is fixed)
    up_positions::Vector{Int}     # Positions of up electrons
    down_positions::Vector{Int}   # Positions of down electrons

    # FSZ configuration
    fsz_config::FSZConfiguration{T}

    # Cached quantities for efficiency
    slater_up::Matrix{T}         # Slater matrix for up electrons
    slater_down::Matrix{T}       # Slater matrix for down electrons

    # Pfaffian/determinant values
    pf_up::T                     # Pfaffian for up sector
    pf_down::T                   # Pfaffian for down sector

    # Green's functions (FSZ-specific)
    green_up::Matrix{T}          # Green's function for up electrons
    green_down::Matrix{T}        # Green's function for down electrons

    # Update-related workspace
    inv_matrix_up::Matrix{T}     # Inverse matrix for up sector
    inv_matrix_down::Matrix{T}   # Inverse matrix for down sector

    # Energy and observables
    local_energy::T
    weight::T
end

"""
    initialize_fsz_state!(state::FSZState{T}, fsz_config::FSZConfiguration{T}) where T

Initialize FSZ state with given configuration.
"""
function initialize_fsz_state!(
    state::FSZState{T},
    fsz_config::FSZConfiguration{T},
) where {T}
    state.fsz_config = fsz_config

    # Initialize electron positions from FSZ configuration
    state.up_positions = copy(fsz_config.up_sites)
    state.down_positions = copy(fsz_config.down_sites)

    # Initialize matrices
    n_up = fsz_config.n_up
    n_down = fsz_config.n_down

    if n_up > 0
        state.slater_up = zeros(T, n_up, n_up)
        state.green_up = zeros(T, n_up, n_up)
        state.inv_matrix_up = zeros(T, n_up, n_up)
    end

    if n_down > 0
        state.slater_down = zeros(T, n_down, n_down)
        state.green_down = zeros(T, n_down, n_down)
        state.inv_matrix_down = zeros(T, n_down, n_down)
    end

    # Initialize Pfaffians
    state.pf_up = one(T)
    state.pf_down = one(T)

    # Initialize energy and weight
    state.local_energy = zero(T)
    state.weight = one(T)

    return state
end

"""
    FSZHamiltonian{T}

Hamiltonian specialized for FSZ calculations.
Precomputes spin-dependent terms for efficiency.
"""
struct FSZHamiltonian{T<:Number}
    # Base Hamiltonian
    base_ham::Hamiltonian{T}

    # FSZ-specific precomputed terms
    up_hop_matrix::SparseMatrixCSC{T,Int}    # Hopping for up electrons
    down_hop_matrix::SparseMatrixCSC{T,Int}  # Hopping for down electrons

    # Interaction terms (spin-separated)
    up_up_coulomb::Vector{Tuple{Int,Int,T}}   # Up-up Coulomb interactions
    down_down_coulomb::Vector{Tuple{Int,Int,T}} # Down-down Coulomb interactions
    up_down_coulomb::Vector{Tuple{Int,Int,T}}   # Up-down Coulomb interactions

    # On-site terms
    onsite_up::Vector{T}          # On-site energies for up electrons
    onsite_down::Vector{T}        # On-site energies for down electrons

    # FSZ configuration reference
    fsz_config::FSZConfiguration{T}
end

"""
    create_fsz_hamiltonian(ham::Hamiltonian{T}, fsz_config::FSZConfiguration{T}) where T

Create FSZ-specialized Hamiltonian from base Hamiltonian and FSZ configuration.
"""
function create_fsz_hamiltonian(
    ham::Hamiltonian{T},
    fsz_config::FSZConfiguration{T},
) where {T}
    n_sites = fsz_config.n_sites

    # Separate hopping terms by spin
    I_up, J_up, V_up = Int[], Int[], T[]
    I_down, J_down, V_down = Int[], Int[], T[]

    for term in ham.transfer_terms
        i, j, spin_i, spin_j, value = term.i, term.j, term.spin_i, term.spin_j, term.value

        if spin_i == spin_j == 0  # Up electrons
            push!(I_up, i)
            push!(J_up, j)
            push!(V_up, value)
        elseif spin_i == spin_j == 1  # Down electrons
            push!(I_down, i)
            push!(J_down, j)
            push!(V_down, value)
        end
    end

    up_hop_matrix = sparse(I_up, J_up, V_up, n_sites, n_sites)
    down_hop_matrix = sparse(I_down, J_down, V_down, n_sites, n_sites)

    # Separate Coulomb interactions by spin combinations
    up_up_coulomb = Tuple{Int,Int,T}[]
    down_down_coulomb = Tuple{Int,Int,T}[]
    up_down_coulomb = Tuple{Int,Int,T}[]

    for term in ham.coulomb_terms
        i, j, value = term.i, term.j, term.value
        spin_i, spin_j = term.spin_i, term.spin_j

        if spin_i == 0 && spin_j == 0
            push!(up_up_coulomb, (i, j, value))
        elseif spin_i == 1 && spin_j == 1
            push!(down_down_coulomb, (i, j, value))
        elseif (spin_i == 0 && spin_j == 1) || (spin_i == 1 && spin_j == 0)
            push!(up_down_coulomb, (i, j, value))
        end
    end

    # On-site energies
    onsite_up = zeros(T, n_sites)
    onsite_down = zeros(T, n_sites)

    # Add chemical potential and other on-site terms
    for term in ham.transfer_terms
        if term.i == term.j  # Diagonal terms
            if term.spin_i == term.spin_j == 0
                onsite_up[term.i] += term.value
            elseif term.spin_i == term.spin_j == 1
                onsite_down[term.i] += term.value
            end
        end
    end

    return FSZHamiltonian{T}(
        ham,
        up_hop_matrix,
        down_hop_matrix,
        up_up_coulomb,
        down_down_coulomb,
        up_down_coulomb,
        onsite_up,
        onsite_down,
        fsz_config,
    )
end

"""
    calculate_fsz_local_energy(state::FSZState{T}, ham::FSZHamiltonian{T}) where T

Calculate local energy for FSZ state.
Optimized version that exploits fixed spin configuration.
"""
function calculate_fsz_local_energy(state::FSZState{T}, ham::FSZHamiltonian{T}) where {T}
    energy = zero(T)

    # Kinetic energy (hopping terms)
    # Up electrons
    if length(state.up_positions) > 0
        for i = 1:length(state.up_positions)
            site_i = state.up_positions[i]
            for j = 1:length(state.up_positions)
                site_j = state.up_positions[j]
                if ham.up_hop_matrix[site_i, site_j] != 0
                    energy += ham.up_hop_matrix[site_i, site_j] * state.green_up[j, i]
                end
            end
        end
    end

    # Down electrons
    if length(state.down_positions) > 0
        for i = 1:length(state.down_positions)
            site_i = state.down_positions[i]
            for j = 1:length(state.down_positions)
                site_j = state.down_positions[j]
                if ham.down_hop_matrix[site_i, site_j] != 0
                    energy += ham.down_hop_matrix[site_i, site_j] * state.green_down[j, i]
                end
            end
        end
    end

    # Interaction energy
    # Up-up interactions
    for (i, j, value) in ham.up_up_coulomb
        n_i_up = i in state.up_positions ? 1 : 0
        n_j_up = j in state.up_positions ? 1 : 0
        energy += value * n_i_up * n_j_up
    end

    # Down-down interactions
    for (i, j, value) in ham.down_down_coulomb
        n_i_down = i in state.down_positions ? 1 : 0
        n_j_down = j in state.down_positions ? 1 : 0
        energy += value * n_i_down * n_j_down
    end

    # Up-down interactions
    for (i, j, value) in ham.up_down_coulomb
        n_i_up = i in state.up_positions ? 1 : 0
        n_j_down = j in state.down_positions ? 1 : 0
        energy += value * n_i_up * n_j_down
    end

    # On-site energy contributions
    for site in state.up_positions
        energy += ham.onsite_up[site]
    end
    for site in state.down_positions
        energy += ham.onsite_down[site]
    end

    state.local_energy = energy
    return energy
end

"""
    fsz_single_electron_update!(state::FSZState{T}, ham::FSZHamiltonian{T},
                                electron_idx::Int, new_site::Int, spin::Int, rng) where T

Perform single electron update in FSZ mode.
Only spatial coordinates change, spin is fixed.
"""
function fsz_single_electron_update!(
    state::FSZState{T},
    ham::FSZHamiltonian{T},
    electron_idx::Int,
    new_site::Int,
    spin::Int,
    rng,
) where {T}
    if spin == 0  # Up electron
        old_site = state.up_positions[electron_idx]

        # Check if the move is valid (new site must be empty or have the same spin)
        fsz_spin = state.fsz_config.spin_config[new_site]
        if fsz_spin != 1 && fsz_spin != 0  # Must be up or empty
            return false, zero(T)  # Reject move
        end

        # Calculate ratio for Slater determinant update
        # This is a simplified version - full implementation would use
        # Sherman-Morrison update for efficiency
        ratio = one(T)  # Placeholder

        # Accept/reject
        if abs(ratio)^2 * rand(rng) > 1.0
            return false, ratio
        end

        # Update position
        state.up_positions[electron_idx] = new_site

        # Update Slater matrix and Green's function (simplified)
        # Full implementation would use efficient matrix updates

        return true, ratio

    else  # Down electron
        old_site = state.down_positions[electron_idx]

        # Check FSZ constraint
        fsz_spin = state.fsz_config.spin_config[new_site]
        if fsz_spin != -1 && fsz_spin != 0  # Must be down or empty
            return false, zero(T)
        end

        # Similar logic for down electrons
        ratio = one(T)  # Placeholder

        if abs(ratio)^2 * rand(rng) > 1.0
            return false, ratio
        end

        state.down_positions[electron_idx] = new_site

        return true, ratio
    end
end

"""
    fsz_two_electron_update!(state::FSZState{T}, ham::FSZHamiltonian{T},
                             electron1_idx::Int, electron2_idx::Int,
                             new_site1::Int, new_site2::Int, spin::Int, rng) where T

Perform two-electron update in FSZ mode.
Specialized for fixed spin configurations.
"""
function fsz_two_electron_update!(
    state::FSZState{T},
    ham::FSZHamiltonian{T},
    electron1_idx::Int,
    electron2_idx::Int,
    new_site1::Int,
    new_site2::Int,
    spin::Int,
    rng,
) where {T}
    # This is a placeholder for the FSZ two-electron update
    # Full implementation would involve:
    # 1. Check FSZ constraints for both moves
    # 2. Calculate Pfaffian ratio using block updates
    # 3. Accept/reject based on probability
    # 4. Update matrices efficiently

    return false, zero(T)  # Placeholder - always reject for now
end

# Export FSZ-related functions and types
export FSZConfiguration,
    FSZState,
    FSZHamiltonian,
    create_fsz_configuration,
    initialize_fsz_state!,
    create_fsz_hamiltonian,
    calculate_fsz_local_energy,
    fsz_single_electron_update!,
    fsz_two_electron_update!
