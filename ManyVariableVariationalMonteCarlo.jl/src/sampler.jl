"""
Monte Carlo sampling engine for ManyVariableVariationalMonteCarlo.jl

Implements the core VMC sampling algorithms including:
- Metropolis-Hastings sampling
- Single and two-electron updates
- Observable measurement
- Statistical analysis
- Integration with wavefunction components

Ported from vmccal.c in the C reference implementation.
"""

using LinearAlgebra
using Random
using Statistics
using StableRNGs

"""
    VMCConfig

Configuration parameters for VMC sampling.
"""
struct VMCConfig
    n_samples::Int
    n_thermalization::Int
    n_measurement::Int
    n_update_per_sample::Int
    acceptance_target::Float64
    temperature::Float64
    use_two_electron_updates::Bool
    two_electron_probability::Float64
end

function VMCConfig(;
    n_samples::Int = 1000,
    n_thermalization::Int = 100,
    n_measurement::Int = 100,
    n_update_per_sample::Int = 1,
    acceptance_target::Float64 = 0.5,
    temperature::Float64 = 1.0,
    use_two_electron_updates::Bool = false,
    two_electron_probability::Float64 = 0.1,
)
    return VMCConfig(
        n_samples,
        n_thermalization,
        n_measurement,
        n_update_per_sample,
        acceptance_target,
        temperature,
        use_two_electron_updates,
        two_electron_probability,
    )
end

"""
    VMCState

Current state of VMC sampling including electron positions and wavefunction values.
"""
mutable struct VMCState{T<:Union{Float64,ComplexF64}}
    # Electron positions
    electron_positions::Vector{Int}
    electron_configuration::Vector{Int}  # Occupation numbers
    n_electrons::Int
    n_sites::Int

    # Wavefunction components
    slater_det::Union{Nothing,Any}  # Will be SlaterDeterminant{T}
    rbm_network::Union{Nothing,Any}  # Will be RBMNetwork{T}
    jastrow_factor::Union{Nothing,Any}  # Will be JastrowFactor{T}
    quantum_projection::Union{Nothing,Any}  # Will be QuantumProjection{T}

    # Current wavefunction value
    wavefunction_value::T
    log_wavefunction_value::Float64

    # Sampling statistics
    n_accepted::Int
    n_rejected::Int
    n_updates::Int

    # Update manager
    update_manager::Union{Nothing,Any}  # Will be UpdateManager{T}

    # Observable manager
    observable_manager::Union{Nothing,Any}  # Will be ObservableManager{T}

    function VMCState{T}(
        n_electrons::Int,
        n_sites::Int,
    ) where {T<:Union{Float64,ComplexF64}}
        electron_positions = zeros(Int, n_electrons)
        electron_configuration = zeros(Int, n_sites)
        new{T}(
            electron_positions,
            electron_configuration,
            n_electrons,
            n_sites,
            nothing,
            nothing,
            nothing,
            nothing,
            zero(T),
            0.0,
            0,
            0,
            0,
            nothing,
            nothing,
        )
    end
end

"""
    VMCResults

Results from VMC sampling including observables and statistics.
"""
struct VMCResults{T<:Union{Float64,ComplexF64}}
    # Energy measurements
    energy_mean::T
    energy_std::T
    energy_samples::Vector{T}

    # Other observables
    observables::Dict{String,Vector{T}}

    # Sampling statistics
    acceptance_rate::Float64
    n_samples::Int
    n_thermalization::Int
    n_measurement::Int

    # Autocorrelation
    autocorrelation_time::Float64
    effective_samples::Int
end

"""
    initialize_vmc_state!(state::VMCState{T}, initial_positions::Vector{Int})

Initialize VMC state with given electron positions.
"""
function initialize_vmc_state!(state::VMCState{T}, initial_positions::Vector{Int}) where {T}
    if length(initial_positions) != state.n_electrons
        throw(ArgumentError("Number of initial positions must match number of electrons"))
    end

    # Set electron positions
    state.electron_positions .= initial_positions

    # Initialize wavefunction value (will be computed by wavefunction components)
    state.wavefunction_value = zero(T)
    state.log_wavefunction_value = 0.0

    # Reset statistics
    state.n_accepted = 0
    state.n_rejected = 0
    state.n_updates = 0
end

"""
    propose_single_electron_move(state::VMCState{T}, rng::AbstractRNG)

Propose a single electron move.
Returns (electron_index, new_position, old_position).
"""
function propose_single_electron_move(state::VMCState{T}, rng::AbstractRNG) where {T}
    # Choose random electron
    electron_idx = rand(rng, 1:state.n_electrons)

    # Choose random new position
    old_position = state.electron_positions[electron_idx]
    new_position = rand(rng, 1:state.n_sites)

    # Avoid same position
    while new_position == old_position
        new_position = rand(rng, 1:state.n_sites)
    end

    return (electron_idx, new_position, old_position)
end

"""
    propose_two_electron_move(state::VMCState{T}, rng::AbstractRNG)

Propose a two electron move.
Returns (electron1_idx, new_pos1, old_pos1, electron2_idx, new_pos2, old_pos2).
"""
function propose_two_electron_move(state::VMCState{T}, rng::AbstractRNG) where {T}
    # Choose two different electrons
    electron1_idx = rand(rng, 1:state.n_electrons)
    electron2_idx = rand(rng, 1:state.n_electrons)

    while electron2_idx == electron1_idx
        electron2_idx = rand(rng, 1:state.n_electrons)
    end

    # Choose new positions
    old_pos1 = state.electron_positions[electron1_idx]
    old_pos2 = state.electron_positions[electron2_idx]

    new_pos1 = rand(rng, 1:state.n_sites)
    new_pos2 = rand(rng, 1:state.n_sites)

    # Avoid same positions
    while new_pos1 == old_pos1
        new_pos1 = rand(rng, 1:state.n_sites)
    end
    while new_pos2 == old_pos2
        new_pos2 = rand(rng, 1:state.n_sites)
    end

    return (electron1_idx, new_pos1, old_pos1, electron2_idx, new_pos2, old_pos2)
end

"""
    compute_wavefunction_ratio(state::VMCState{T}, move_type::Symbol, move_data)

Compute wavefunction ratio for proposed move.
"""
function compute_wavefunction_ratio(
    state::VMCState{T},
    move_type::Symbol,
    move_data,
) where {T}
    if move_type == :single_electron
        electron_idx, new_pos, old_pos = move_data

        # Compute ratio from Slater determinant
        if state.slater_det !== nothing
            slater_ratio =
                compute_slater_ratio(state.slater_det, electron_idx, new_pos, old_pos)
        else
            slater_ratio = one(T)
        end

        # Compute ratio from RBM
        if state.rbm_network !== nothing
            rbm_ratio = compute_rbm_ratio(state.rbm_network, electron_idx, new_pos, old_pos)
        else
            rbm_ratio = one(T)
        end

        return slater_ratio * rbm_ratio

    elseif move_type == :two_electron
        electron1_idx, new_pos1, old_pos1, electron2_idx, new_pos2, old_pos2 = move_data

        # Compute ratio from Slater determinant
        if state.slater_det !== nothing
            slater_ratio = compute_slater_ratio_two_electron(
                state.slater_det,
                electron1_idx,
                new_pos1,
                old_pos1,
                electron2_idx,
                new_pos2,
                old_pos2,
            )
        else
            slater_ratio = one(T)
        end

        # Compute ratio from RBM
        if state.rbm_network !== nothing
            rbm_ratio = compute_rbm_ratio_two_electron(
                state.rbm_network,
                electron1_idx,
                new_pos1,
                old_pos1,
                electron2_idx,
                new_pos2,
                old_pos2,
            )
        else
            rbm_ratio = one(T)
        end

        return slater_ratio * rbm_ratio
    else
        throw(ArgumentError("Unknown move type: $move_type"))
    end
end

"""
    compute_slater_ratio(slater_det, electron_idx::Int, new_pos::Int, old_pos::Int)

Compute Slater determinant ratio for single electron move.
"""
function compute_slater_ratio(slater_det, electron_idx::Int, new_pos::Int, old_pos::Int)
    # This would call the appropriate method from the SlaterDeterminant type
    # For now, return 1.0 as placeholder
    return 1.0
end

"""
    compute_slater_ratio_two_electron(slater_det, electron1_idx::Int, new_pos1::Int, old_pos1::Int,
                                     electron2_idx::Int, new_pos2::Int, old_pos2::Int)

Compute Slater determinant ratio for two electron move.
"""
function compute_slater_ratio_two_electron(
    slater_det,
    electron1_idx::Int,
    new_pos1::Int,
    old_pos1::Int,
    electron2_idx::Int,
    new_pos2::Int,
    old_pos2::Int,
)
    # This would call the appropriate method from the SlaterDeterminant type
    # For now, return 1.0 as placeholder
    return 1.0
end

"""
    compute_rbm_ratio(rbm_network, electron_idx::Int, new_pos::Int, old_pos::Int)

Compute RBM ratio for single electron move.
"""
function compute_rbm_ratio(rbm_network, electron_idx::Int, new_pos::Int, old_pos::Int)
    # This would call the appropriate method from the RBMNetwork type
    # For now, return 1.0 as placeholder
    return 1.0
end

"""
    compute_rbm_ratio_two_electron(rbm_network, electron1_idx::Int, new_pos1::Int, old_pos1::Int,
                                  electron2_idx::Int, new_pos2::Int, old_pos2::Int)

Compute RBM ratio for two electron move.
"""
function compute_rbm_ratio_two_electron(
    rbm_network,
    electron1_idx::Int,
    new_pos1::Int,
    old_pos1::Int,
    electron2_idx::Int,
    new_pos2::Int,
    old_pos2::Int,
)
    # This would call the appropriate method from the RBMNetwork type
    # For now, return 1.0 as placeholder
    return 1.0
end

"""
    accept_move!(state::VMCState{T}, move_type::Symbol, move_data)

Accept the proposed move and update state.
"""
function accept_move!(state::VMCState{T}, move_type::Symbol, move_data) where {T}
    if move_type == :single_electron
        electron_idx, new_pos, old_pos = move_data
        state.electron_positions[electron_idx] = new_pos

        # Update wavefunction components
        if state.slater_det !== nothing
            update_slater_single_electron!(state.slater_det, electron_idx, new_pos, old_pos)
        end
        if state.rbm_network !== nothing
            update_rbm_single_electron!(state.rbm_network, electron_idx, new_pos, old_pos)
        end

    elseif move_type == :two_electron
        electron1_idx, new_pos1, old_pos1, electron2_idx, new_pos2, old_pos2 = move_data
        state.electron_positions[electron1_idx] = new_pos1
        state.electron_positions[electron2_idx] = new_pos2

        # Update wavefunction components
        if state.slater_det !== nothing
            update_slater_two_electron!(
                state.slater_det,
                electron1_idx,
                new_pos1,
                old_pos1,
                electron2_idx,
                new_pos2,
                old_pos2,
            )
        end
        if state.rbm_network !== nothing
            update_rbm_two_electron!(
                state.rbm_network,
                electron1_idx,
                new_pos1,
                old_pos1,
                electron2_idx,
                new_pos2,
                old_pos2,
            )
        end
    end

    state.n_accepted += 1
    state.n_updates += 1
end

"""
    reject_move!(state::VMCState{T})

Reject the proposed move.
"""
function reject_move!(state::VMCState{T}) where {T}
    state.n_rejected += 1
    state.n_updates += 1
end

"""
    update_slater_single_electron!(slater_det, electron_idx::Int, new_pos::Int, old_pos::Int)

Update Slater determinant after single electron move.
"""
function update_slater_single_electron!(
    slater_det,
    electron_idx::Int,
    new_pos::Int,
    old_pos::Int,
)
    # This would call the appropriate method from the SlaterDeterminant type
    # For now, do nothing as placeholder
end

"""
    update_slater_two_electron!(slater_det, electron1_idx::Int, new_pos1::Int, old_pos1::Int,
                               electron2_idx::Int, new_pos2::Int, old_pos2::Int)

Update Slater determinant after two electron move.
"""
function update_slater_two_electron!(
    slater_det,
    electron1_idx::Int,
    new_pos1::Int,
    old_pos1::Int,
    electron2_idx::Int,
    new_pos2::Int,
    old_pos2::Int,
)
    # This would call the appropriate method from the SlaterDeterminant type
    # For now, do nothing as placeholder
end

"""
    update_rbm_single_electron!(rbm_network, electron_idx::Int, new_pos::Int, old_pos::Int)

Update RBM after single electron move.
"""
function update_rbm_single_electron!(
    rbm_network,
    electron_idx::Int,
    new_pos::Int,
    old_pos::Int,
)
    # This would call the appropriate method from the RBMNetwork type
    # For now, do nothing as placeholder
end

"""
    update_rbm_two_electron!(rbm_network, electron1_idx::Int, new_pos1::Int, old_pos1::Int,
                            electron2_idx::Int, new_pos2::Int, old_pos2::Int)

Update RBM after two electron move.
"""
function update_rbm_two_electron!(
    rbm_network,
    electron1_idx::Int,
    new_pos1::Int,
    old_pos1::Int,
    electron2_idx::Int,
    new_pos2::Int,
    old_pos2::Int,
)
    # This would call the appropriate method from the RBMNetwork type
    # For now, do nothing as placeholder
end

"""
    metropolis_step!(state::VMCState{T}, config::VMCConfig, rng::AbstractRNG)

Perform one Metropolis step.
"""
function metropolis_step!(state::VMCState{T}, config::VMCConfig, rng::AbstractRNG) where {T}
    # Choose move type
    if config.use_two_electron_updates && rand(rng) < config.two_electron_probability
        move_type = :two_electron
        move_data = propose_two_electron_move(state, rng)
    else
        move_type = :single_electron
        move_data = propose_single_electron_move(state, rng)
    end

    # Compute wavefunction ratio
    ratio = compute_wavefunction_ratio(state, move_type, move_data)

    # Metropolis acceptance criterion
    if abs(ratio)^2 > rand(rng)
        accept_move!(state, move_type, move_data)
    else
        reject_move!(state)
    end
end

"""
    measure_energy(state::VMCState{T})

Measure energy for current state.
"""
function measure_energy(state::VMCState{T}) where {T}
    # This would compute the energy for the current electron configuration
    # For now, return a placeholder value
    return zero(T)
end

"""
    measure_observables(state::VMCState{T})

Measure all observables for current state.
"""
function measure_observables(state::VMCState{T}) where {T}
    observables = Dict{String,T}()

    # Measure energy
    observables["energy"] = measure_energy(state)

    # Add other observables as needed

    return observables
end

"""
    run_vmc_sampling!(state::VMCState{T}, config::VMCConfig, rng::AbstractRNG)

Run VMC sampling with given configuration.
"""
function run_vmc_sampling!(
    state::VMCState{T},
    config::VMCConfig,
    rng::AbstractRNG,
) where {T}
    # Thermalization phase
    for i = 1:config.n_thermalization
        for j = 1:config.n_update_per_sample
            metropolis_step!(state, config, rng)
        end
    end

    # Measurement phase
    energy_samples = Vector{T}()
    observable_samples = Dict{String,Vector{T}}()

    for i = 1:config.n_measurement
        for j = 1:config.n_update_per_sample
            metropolis_step!(state, config, rng)
        end

        # Measure observables
        observables = measure_observables(state)
        push!(energy_samples, observables["energy"])

        for (key, value) in observables
            if !haskey(observable_samples, key)
                observable_samples[key] = Vector{T}()
            end
            push!(observable_samples[key], value)
        end
    end

    # Compute statistics
    energy_mean = mean(energy_samples)
    energy_std = std(energy_samples)
    acceptance_rate = state.n_accepted / state.n_updates

    # Compute autocorrelation time (simplified)
    autocorrelation_time = 1.0  # Placeholder
    effective_samples = length(energy_samples)  # Placeholder

    return VMCResults{T}(
        energy_mean,
        energy_std,
        energy_samples,
        observable_samples,
        acceptance_rate,
        config.n_measurement,
        config.n_thermalization,
        config.n_measurement,
        autocorrelation_time,
        effective_samples,
    )
end

# Enhanced VMC sampling with integrated components

"""
    EnhancedVMCState{T}

Enhanced VMC state with integrated wavefunction components and update algorithms.
"""
mutable struct EnhancedVMCState{T<:Union{Float64,ComplexF64}}
    # Electron configuration
    electron_positions::Vector{Int}
    electron_configuration::Vector{Int}
    n_electrons::Int
    n_sites::Int

    # Wavefunction components
    slater_det::Union{Nothing,Any}
    rbm_network::Union{Nothing,Any}
    jastrow_factor::Union{Nothing,Any}
    quantum_projection::Union{Nothing,Any}

    # Monte Carlo components
    update_manager::Union{Nothing,Any}
    observable_manager::Union{Nothing,Any}

    # Current wavefunction value
    wavefunction_value::T
    log_wavefunction_value::Float64

    # Sampling statistics
    n_accepted::Int
    n_rejected::Int
    n_updates::Int
    total_energy::T
    energy_variance::T

    function EnhancedVMCState{T}(n_electrons::Int, n_sites::Int) where {T}
        electron_positions = zeros(Int, n_electrons)
        electron_configuration = zeros(Int, n_sites)
        new{T}(
            electron_positions,
            electron_configuration,
            n_electrons,
            n_sites,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            zero(T),
            0.0,
            0,
            0,
            0,
            zero(T),
            zero(T),
        )
    end
end

"""
    initialize_enhanced_vmc_state!(state::EnhancedVMCState{T}, initial_positions::Vector{Int},
                                  slater_det, rbm_network, jastrow_factor, quantum_projection,
                                  update_manager, observable_manager) where T

Initialize enhanced VMC state with all components.
"""
function initialize_enhanced_vmc_state!(
    state::EnhancedVMCState{T},
    initial_positions::Vector{Int},
    slater_det,
    rbm_network,
    jastrow_factor,
    quantum_projection,
    update_manager,
    observable_manager,
) where {T}
    if length(initial_positions) != state.n_electrons
        throw(ArgumentError("Number of initial positions must match number of electrons"))
    end

    # Set electron positions and configuration
    state.electron_positions .= initial_positions
    fill!(state.electron_configuration, 0)
    for pos in initial_positions
        state.electron_configuration[pos] = 1
    end

    # Set wavefunction components
    state.slater_det = slater_det
    state.rbm_network = rbm_network
    state.jastrow_factor = jastrow_factor
    state.quantum_projection = quantum_projection

    # Set Monte Carlo components
    state.update_manager = update_manager
    state.observable_manager = observable_manager

    # Initialize wavefunction value
    state.wavefunction_value = compute_total_wavefunction_value(state)
    state.log_wavefunction_value = log(abs(state.wavefunction_value))

    # Reset statistics
    state.n_accepted = 0
    state.n_rejected = 0
    state.n_updates = 0
    state.total_energy = zero(T)
    state.energy_variance = zero(T)
end

"""
    compute_total_wavefunction_value(state::EnhancedVMCState{T}) where T

Compute total wavefunction value from all components.
"""
function compute_total_wavefunction_value(state::EnhancedVMCState{T}) where {T}
    total_value = one(T)

    # Slater determinant contribution
    if state.slater_det !== nothing
        slater_value = get_determinant_value(state.slater_det)
        total_value *= slater_value
    end

    # RBM contribution
    if state.rbm_network !== nothing
        rbm_value = rbm_weight(
            state.rbm_network,
            state.electron_configuration,
            zeros(Int, state.n_sites),
        )  # Simplified hidden state
        total_value *= rbm_value
    end

    # Jastrow factor contribution
    if state.jastrow_factor !== nothing
        jastrow_value = jastrow_factor(
            state.jastrow_factor,
            state.electron_positions,
            state.electron_configuration,
            state.electron_configuration,
        )
        total_value *= jastrow_value
    end

    # Quantum projection contribution
    if state.quantum_projection !== nothing
        projection_value = calculate_projection_ratio(
            state.quantum_projection,
            state.electron_positions,
            state.electron_configuration,
            state.electron_configuration,
        )
        total_value *= projection_value
    end

    return total_value
end

"""
    propose_enhanced_move(state::EnhancedVMCState{T}, rng::AbstractRNG) where T

Propose a move using the update manager.
"""
function propose_enhanced_move(state::EnhancedVMCState{T}, rng::AbstractRNG) where {T}
    if state.update_manager === nothing
        throw(ArgumentError("Update manager not initialized"))
    end

    return propose_update(
        state.update_manager,
        state.electron_positions,
        state.electron_configuration,
        rng,
    )
end

"""
    compute_move_ratio(state::EnhancedVMCState{T}, result::UpdateResult{T}) where T

Compute wavefunction ratio for proposed move.
"""
function compute_move_ratio(state::EnhancedVMCState{T}, result::UpdateResult{T}) where {T}
    if !result.success
        return zero(T)
    end

    # Create temporary state for ratio calculation
    temp_positions = copy(state.electron_positions)
    temp_configuration = copy(state.electron_configuration)

    # Apply move to temporary state
    if result.update_type == SINGLE_ELECTRON
        ele_idx = result.electron_indices[1]
        old_site = result.site_indices[1]
        new_site = result.site_indices[2]

        temp_positions[ele_idx] = new_site
        temp_configuration[old_site] = 0
        temp_configuration[new_site] = 1

    elseif result.update_type == TWO_ELECTRON
        ele1_idx = result.electron_indices[1]
        ele2_idx = result.electron_indices[2]
        old_site1 = result.site_indices[1]
        old_site2 = result.site_indices[2]
        new_site1 = result.site_indices[3]
        new_site2 = result.site_indices[4]

        temp_positions[ele1_idx] = new_site1
        temp_positions[ele2_idx] = new_site2
        temp_configuration[old_site1] = 0
        temp_configuration[old_site2] = 0
        temp_configuration[new_site1] = 1
        temp_configuration[new_site2] = 1
    end

    # Compute wavefunction value for new configuration
    new_value = one(T)

    # Slater determinant ratio
    if state.slater_det !== nothing
        if result.update_type == SINGLE_ELECTRON
            ele_idx = result.electron_indices[1]
            old_site = result.site_indices[1]
            new_site = result.site_indices[2]
            slater_ratio = update_slater!(state.slater_det, ele_idx, new_site, one(T))
        else
            slater_ratio = one(T)  # Simplified for two-electron moves
        end
        new_value *= slater_ratio
    end

    # RBM ratio
    if state.rbm_network !== nothing
        rbm_ratio =
            rbm_weight(state.rbm_network, temp_configuration, zeros(Int, state.n_sites))
        new_value *= rbm_ratio
    end

    # Jastrow factor ratio
    if state.jastrow_factor !== nothing
        jastrow_ratio = jastrow_ratio(
            state.jastrow_factor,
            state.electron_positions,
            state.electron_configuration,
            state.electron_configuration,
            temp_positions,
            temp_configuration,
            temp_configuration,
        )
        new_value *= jastrow_ratio
    end

    # Quantum projection ratio
    if state.quantum_projection !== nothing
        projection_ratio = calculate_projection_ratio(
            state.quantum_projection,
            temp_positions,
            temp_configuration,
            temp_configuration,
        )
        new_value *= projection_ratio
    end

    return new_value / state.wavefunction_value
end

"""
    accept_enhanced_move!(state::EnhancedVMCState{T}, result::UpdateResult{T}) where T

Accept the proposed move and update all components.
"""
function accept_enhanced_move!(
    state::EnhancedVMCState{T},
    result::UpdateResult{T},
) where {T}
    if !result.success
        return
    end

    # Update electron positions and configuration
    if result.update_type == SINGLE_ELECTRON
        ele_idx = result.electron_indices[1]
        old_site = result.site_indices[1]
        new_site = result.site_indices[2]

        state.electron_positions[ele_idx] = new_site
        state.electron_configuration[old_site] = 0
        state.electron_configuration[new_site] = 1

    elseif result.update_type == TWO_ELECTRON
        ele1_idx = result.electron_indices[1]
        ele2_idx = result.electron_indices[2]
        old_site1 = result.site_indices[1]
        old_site2 = result.site_indices[2]
        new_site1 = result.site_indices[3]
        new_site2 = result.site_indices[4]

        state.electron_positions[ele1_idx] = new_site1
        state.electron_positions[ele2_idx] = new_site2
        state.electron_configuration[old_site1] = 0
        state.electron_configuration[old_site2] = 0
        state.electron_configuration[new_site1] = 1
        state.electron_configuration[new_site2] = 1
    end

    # Update wavefunction value
    state.wavefunction_value = compute_total_wavefunction_value(state)
    state.log_wavefunction_value = log(abs(state.wavefunction_value))

    # Update statistics
    state.n_accepted += 1
    state.n_updates += 1

    # Update update manager statistics
    if state.update_manager !== nothing
        accept_update!(state.update_manager, result)
    end
end

"""
    reject_enhanced_move!(state::EnhancedVMCState{T}, result::UpdateResult{T}) where T

Reject the proposed move and update statistics.
"""
function reject_enhanced_move!(
    state::EnhancedVMCState{T},
    result::UpdateResult{T},
) where {T}
    state.n_rejected += 1
    state.n_updates += 1

    # Update update manager statistics
    if state.update_manager !== nothing
        reject_update!(state.update_manager, result)
    end
end

"""
    enhanced_metropolis_step!(state::EnhancedVMCState{T}, rng::AbstractRNG) where T

Perform one enhanced Metropolis step.
"""
function enhanced_metropolis_step!(state::EnhancedVMCState{T}, rng::AbstractRNG) where {T}
    # Propose move
    result = propose_enhanced_move(state, rng)

    if !result.success
        return
    end

    # Compute move ratio
    ratio = compute_move_ratio(state, result)

    # Metropolis acceptance criterion
    if abs(ratio)^2 > rand(rng)
        accept_enhanced_move!(state, result)
    else
        reject_enhanced_move!(state, result)
    end
end

"""
    run_enhanced_vmc_sampling!(state::EnhancedVMCState{T}, config::VMCConfig, rng::AbstractRNG) where T

Run enhanced VMC sampling with integrated components.
"""
function run_enhanced_vmc_sampling!(
    state::EnhancedVMCState{T},
    config::VMCConfig,
    rng::AbstractRNG,
) where {T}
    # Thermalization phase
    for i = 1:config.n_thermalization
        for j = 1:config.n_update_per_sample
            enhanced_metropolis_step!(state, rng)
        end
    end

    # Measurement phase
    energy_samples = Vector{T}()
    observable_samples = Dict{String,Vector{T}}()

    for i = 1:config.n_measurement
        for j = 1:config.n_update_per_sample
            enhanced_metropolis_step!(state, rng)
        end

        # Measure observables using observable manager
        if state.observable_manager !== nothing
            measure_observables!(
                state.observable_manager,
                state.electron_positions,
                state.electron_configuration,
                1.0,
            )
        end

        # Get energy from observable manager
        if state.observable_manager !== nothing
            energy_stats = get_observable_statistics(state.observable_manager, "energy")
            push!(energy_samples, energy_stats.mean)
        else
            # Fallback to simple energy calculation
            energy = measure_energy(state)
            push!(energy_samples, energy)
        end
    end

    # Compute statistics
    energy_mean = mean(energy_samples)
    energy_std = std(energy_samples)
    acceptance_rate = state.n_accepted / state.n_updates

    # Compute autocorrelation time (simplified)
    autocorrelation_time = 1.0  # Placeholder
    effective_samples = length(energy_samples)  # Placeholder

    return VMCResults{T}(
        energy_mean,
        energy_std,
        energy_samples,
        observable_samples,
        acceptance_rate,
        config.n_measurement,
        config.n_thermalization,
        config.n_measurement,
        autocorrelation_time,
        effective_samples,
    )
end

"""
    get_acceptance_rate(state::VMCState{T})

Get current acceptance rate.
"""
function get_acceptance_rate(state::VMCState{T}) where {T}
    if state.n_updates == 0
        return 0.0
    end
    return state.n_accepted / state.n_updates
end

"""
    reset_vmc_state!(state::VMCState{T})

Reset VMC state statistics.
"""
function reset_vmc_state!(state::VMCState{T}) where {T}
    state.n_accepted = 0
    state.n_rejected = 0
    state.n_updates = 0
end
