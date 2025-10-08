# Backflow corrections for wavefunction
# Based on mVMC C implementation: SlaterElmBF_*, CalculateMAll_BF_*, etc.

"""
    BackflowConfiguration{T}

Configuration for backflow corrections to the wavefunction.
Backflow transformations modify the single-particle orbitals based on
the positions of other electrons, leading to improved variational wavefunctions.
"""
struct BackflowConfiguration{T<:Number}
    # Backflow parameters
    eta_parameters::Matrix{T}     # η parameters for backflow transformation
    n_orbitals::Int              # Number of orbitals
    n_particles::Int             # Number of particles

    # Range and indexing
    n_backflow_idx::Int          # Number of backflow indices
    backflow_idx::Vector{Int}    # Backflow indices
    range_idx::Vector{Int}       # Range indices for backflow
    pos_bf::Vector{Int}          # Position indices for backflow

    # Submatrix indexing for efficiency
    bf_sub_idx::Vector{Vector{Int}}  # Backflow submatrix indices
    n_bf_idx_total::Int             # Total number of BF indices
    n_range_idx::Int                # Number of range indices

    # Flags and options
    eta_flag::Bool               # Whether to use eta parameters
    smp_eta_flag::Bool          # Sample eta parameters
    use_real::Bool              # Use real arithmetic when possible
end

"""
    create_backflow_configuration(n_orbitals::Int, n_particles::Int;
                                 eta_flag=true, smp_eta_flag=false, use_real=false)

Create backflow configuration with default parameters.

C実装参考: slater.c 1行目から1506行目まで
"""
function create_backflow_configuration(
    n_orbitals::Int,
    n_particles::Int;
    eta_flag = true,
    smp_eta_flag = false,
    use_real = false,
)
    T = use_real ? Float64 : ComplexF64

    # Initialize eta parameters (simplified - should be optimized)
    eta_parameters = zeros(T, n_orbitals, n_particles)

    # Initialize indices (simplified)
    n_backflow_idx = n_orbitals
    backflow_idx = collect(1:n_backflow_idx)
    range_idx = collect(1:n_particles)
    pos_bf = collect(1:n_particles)

    # Submatrix indexing
    bf_sub_idx = [collect(1:n_orbitals) for _ = 1:n_particles]
    n_bf_idx_total = n_backflow_idx
    n_range_idx = n_particles

    return BackflowConfiguration{T}(
        eta_parameters,
        n_orbitals,
        n_particles,
        n_backflow_idx,
        backflow_idx,
        range_idx,
        pos_bf,
        bf_sub_idx,
        n_bf_idx_total,
        n_range_idx,
        eta_flag,
        smp_eta_flag,
        use_real,
    )
end

"""
    BackflowState{T}

State for backflow-corrected wavefunction calculations.
Contains the transformed orbitals and intermediate quantities.
"""
mutable struct BackflowState{T<:Number}
    # Original and transformed matrices
    original_slater::Matrix{T}    # Original Slater matrix
    bf_slater::Matrix{T}         # Backflow-corrected Slater matrix

    # Backflow transformation matrices
    bf_matrix::Matrix{T}         # Backflow transformation matrix
    bf_gradient::Array{T,3}      # Gradient of backflow transformation

    # Intermediate quantities for efficiency
    m_all::Matrix{T}             # All M matrices for backflow
    new_pf_m_bf::Vector{T}       # New Pfaffian M for backflow

    # Update-related workspace
    delta_matrix::Matrix{T}      # Delta matrix for updates
    inv_delta::Matrix{T}         # Inverse delta matrix

    # Configuration reference
    bf_config::BackflowConfiguration{T}

    # Electron positions (needed for backflow)
    positions::Vector{Int}

    # Cached quantities
    pf_ratio::T                  # Pfaffian ratio for updates
    bf_weight::T                 # Backflow weight contribution
end

"""
    initialize_backflow_state!(state::BackflowState{T}, bf_config::BackflowConfiguration{T},
                              positions::Vector{Int}) where T

Initialize backflow state with given configuration and electron positions.
"""
function initialize_backflow_state!(
    state::BackflowState{T},
    bf_config::BackflowConfiguration{T},
    positions::Vector{Int},
) where {T}
    n_orb = bf_config.n_orbitals
    n_part = bf_config.n_particles

    # Initialize matrices
    state.original_slater = zeros(T, n_part, n_orb)
    state.bf_slater = zeros(T, n_part, n_orb)
    state.bf_matrix = zeros(T, n_orb, n_orb)
    state.bf_gradient = zeros(T, n_orb, n_orb, n_part)

    state.m_all = zeros(T, n_part, n_part)
    state.new_pf_m_bf = zeros(T, n_part)

    state.delta_matrix = zeros(T, n_part, n_part)
    state.inv_delta = zeros(T, n_part, n_part)

    state.bf_config = bf_config
    state.positions = copy(positions)

    state.pf_ratio = one(T)
    state.bf_weight = one(T)

    return state
end

"""
    calculate_backflow_matrix!(state::BackflowState{T}) where T

Calculate the backflow transformation matrix.
This transforms the single-particle orbitals based on many-body correlations.
"""
function calculate_backflow_matrix!(state::BackflowState{T}) where {T}
    bf_config = state.bf_config
    positions = state.positions

    # Reset backflow matrix
    fill!(state.bf_matrix, zero(T))

    # Add identity (unperturbed orbitals)
    for i = 1:bf_config.n_orbitals
        state.bf_matrix[i, i] = one(T)
    end

    if !bf_config.eta_flag
        return  # No backflow correction
    end

    # Apply backflow transformation
    for i = 1:bf_config.n_particles
        pos_i = positions[i]
        for j = 1:bf_config.n_particles
            if i != j
                pos_j = positions[j]
                for orb = 1:bf_config.n_orbitals
                    # Simplified backflow transformation
                    # Real implementation would use distance-dependent kernels
                    eta_val = bf_config.eta_parameters[orb, i]
                    state.bf_matrix[orb, pos_j] += eta_val
                end
            end
        end
    end
end

"""
    apply_backflow_correction!(state::BackflowState{T}) where T

Apply backflow correction to the Slater matrix.
"""
function apply_backflow_correction!(state::BackflowState{T}) where {T}
    calculate_backflow_matrix!(state)

    # Apply transformation: Ψ_BF = Ψ_0 × BF_matrix
    state.bf_slater = state.original_slater * state.bf_matrix

    return state.bf_slater
end

"""
    calculate_backflow_gradient!(state::BackflowState{T}, particle_idx::Int) where T

Calculate gradient of backflow transformation with respect to particle position.
Needed for efficient updates.
"""
function calculate_backflow_gradient!(state::BackflowState{T}, particle_idx::Int) where {T}
    bf_config = state.bf_config
    positions = state.positions

    # Reset gradient for this particle
    state.bf_gradient[:, :, particle_idx] .= zero(T)

    if !bf_config.eta_flag
        return
    end

    # Calculate gradient
    for orb = 1:bf_config.n_orbitals
        for other_particle = 1:bf_config.n_particles
            if other_particle != particle_idx
                # Gradient of backflow correction
                eta_val = bf_config.eta_parameters[orb, particle_idx]
                pos_other = positions[other_particle]
                state.bf_gradient[orb, pos_other, particle_idx] += eta_val
            end
        end
    end
end

"""
    calculate_backflow_ratio(state::BackflowState{T}, particle_idx::Int, new_position::Int) where T

Calculate the ratio of wavefunctions for a proposed particle move with backflow.
"""
function calculate_backflow_ratio(
    state::BackflowState{T},
    particle_idx::Int,
    new_position::Int,
) where {T}
    old_position = state.positions[particle_idx]

    if old_position == new_position
        return one(T)
    end

    # Calculate change in backflow matrix due to the move
    delta_bf = zeros(T, state.bf_config.n_orbitals, state.bf_config.n_orbitals)

    if state.bf_config.eta_flag
        for orb = 1:state.bf_config.n_orbitals
            for other_particle = 1:state.bf_config.n_particles
                if other_particle != particle_idx
                    eta_val = state.bf_config.eta_parameters[orb, particle_idx]
                    pos_other = state.positions[other_particle]

                    # Remove old contribution
                    delta_bf[orb, old_position] -= eta_val
                    # Add new contribution
                    delta_bf[orb, new_position] += eta_val
                end
            end
        end
    end

    # Calculate ratio using Sherman-Morrison-like update
    # This is a simplified version - full implementation would be more complex
    ratio = one(T)

    # Apply delta to get new Slater matrix
    new_bf_slater = state.bf_slater + state.original_slater * delta_bf

    # Calculate determinant ratio (simplified)
    # Real implementation would use efficient determinant ratio calculation
    old_det = det(state.bf_slater)
    new_det = det(new_bf_slater)

    if abs(old_det) > 1e-12
        ratio = new_det / old_det
    end

    return ratio
end

"""
    update_backflow_state!(state::BackflowState{T}, particle_idx::Int, new_position::Int) where T

Update backflow state after accepting a particle move.
"""
function update_backflow_state!(
    state::BackflowState{T},
    particle_idx::Int,
    new_position::Int,
) where {T}
    # Update position
    state.positions[particle_idx] = new_position

    # Recalculate backflow matrix and apply correction
    apply_backflow_correction!(state)

    # Update gradients if needed
    calculate_backflow_gradient!(state, particle_idx)

    return state
end

"""
    calculate_backflow_energy_contribution(state::BackflowState{T}, hamiltonian) where T

Calculate energy contribution from backflow corrections.
"""
function calculate_backflow_energy_contribution(
    state::BackflowState{T},
    hamiltonian,
) where {T}
    # Energy contribution from backflow is typically included in the
    # kinetic energy through the modified orbitals
    # This is a placeholder for more sophisticated implementations

    energy_contribution = zero(T)

    # The backflow energy comes from the kinetic energy operator
    # acting on the backflow-corrected wavefunction
    # This requires careful treatment of derivatives

    return energy_contribution
end

"""
    optimize_backflow_parameters!(bf_config::BackflowConfiguration{T},
                                 gradient::Matrix{T}, step_size::Real) where T

Update backflow parameters using computed gradients.
"""
function optimize_backflow_parameters!(
    bf_config::BackflowConfiguration{T},
    gradient::Matrix{T},
    step_size::Real,
) where {T}
    if bf_config.smp_eta_flag
        # Update eta parameters
        bf_config.eta_parameters .-= step_size .* gradient
    end
end

"""
    calculate_bf_slater_element(state::BackflowState{T}, i::Int, j::Int) where T

Calculate individual element of backflow-corrected Slater matrix.
Equivalent to SlaterElmBF_fcmp/SlaterElmBF_real in mVMC.
"""
function calculate_bf_slater_element(state::BackflowState{T}, i::Int, j::Int) where {T}
    # This calculates ⟨φᵢ|Ψ_BF⟩ where φᵢ is a single-particle orbital
    # and Ψ_BF is the backflow-corrected many-body wavefunction

    element = zero(T)

    # Sum over all configurations with particle i at orbital j
    for k = 1:state.bf_config.n_orbitals
        element += state.original_slater[i, k] * state.bf_matrix[k, j]
    end

    return element
end

# Export backflow-related functions and types
export BackflowConfiguration,
    BackflowState,
    create_backflow_configuration,
    initialize_backflow_state!,
    calculate_backflow_matrix!,
    apply_backflow_correction!,
    calculate_backflow_gradient!,
    calculate_backflow_ratio,
    update_backflow_state!,
    calculate_backflow_energy_contribution,
    optimize_backflow_parameters!,
    calculate_bf_slater_element
