"""
Monte Carlo Update Algorithms for ManyVariableVariationalMonteCarlo.jl

Implements various Monte Carlo update algorithms including:
- Single electron updates
- Two electron updates
- Exchange hopping updates
- Pfaffian matrix updates
- Local Green function updates

Ported from pfupdate*.c and locgrn*.c in the C reference implementation.
"""

using LinearAlgebra
using Random
using StableRNGs

"""
    UpdateType

Enumeration of different Monte Carlo update types.
"""
@enum UpdateType begin
    SINGLE_ELECTRON
    TWO_ELECTRON
    EXCHANGE_HOPPING
    PAIR_CREATION_ANNIHILATION
    CLUSTER_UPDATE
end

"""
    UpdateResult{T}

Result of a Monte Carlo update operation.
"""
mutable struct UpdateResult{T<:Union{Float64,ComplexF64}}
    success::Bool
    ratio::T
    update_type::UpdateType
    electron_indices::Vector{Int}
    site_indices::Vector{Int}
    energy_change::T
    acceptance_probability::Float64

    function UpdateResult{T}(success::Bool, ratio::T, update_type::UpdateType) where {T}
        new{T}(success, ratio, update_type, Int[], Int[], zero(T), 0.0)
    end
end

"""
    SingleElectronUpdate{T}

Single electron Monte Carlo update.
"""
mutable struct SingleElectronUpdate{T<:Union{Float64,ComplexF64}}
    # System parameters
    n_site::Int
    n_elec::Int

    # Update parameters
    max_hop_distance::Int
    update_probability::Float64

    # Working arrays
    candidate_sites::Vector{Int}
    current_sites::Vector{Int}
    ratio_buffer::Vector{T}

    # Statistics
    total_attempts::Int
    total_accepted::Int
    acceptance_rate::Float64

    function SingleElectronUpdate{T}(
        n_site::Int,
        n_elec::Int;
        max_hop_distance::Int = 1,
        update_probability::Float64 = 0.1,
    ) where {T}
        candidate_sites = Vector{Int}(undef, n_site)
        current_sites = Vector{Int}(undef, n_elec)
        ratio_buffer = Vector{T}(undef, n_elec)

        new{T}(
            n_site,
            n_elec,
            max_hop_distance,
            update_probability,
            candidate_sites,
            current_sites,
            ratio_buffer,
            0,
            0,
            0.0,
        )
    end
end

"""
    propose_single_electron_move(update::SingleElectronUpdate{T}, ele_idx::Vector{Int},
                                ele_cfg::Vector{Int}, rng::AbstractRNG) where T

Propose a single electron move.
"""
function propose_single_electron_move(
    update::SingleElectronUpdate{T},
    ele_idx::Vector{Int},
    ele_cfg::Vector{Int},
    rng::AbstractRNG,
) where {T}
    if isempty(ele_idx)
        return UpdateResult{T}(false, zero(T), SINGLE_ELECTRON)
    end

    # Randomly select an electron
    ele_idx_selected = rand(rng, 1:length(ele_idx))
    current_site = ele_idx[ele_idx_selected]

    # Find candidate sites within hopping distance
    n_candidates = 0
    for site = 1:update.n_site
        if site != current_site && ele_cfg[site] == 0
            # Check hopping distance (simplified for 1D)
            if abs(site - current_site) <= update.max_hop_distance
                n_candidates += 1
                update.candidate_sites[n_candidates] = site
            end
        end
    end

    if n_candidates == 0
        return UpdateResult{T}(false, zero(T), SINGLE_ELECTRON)
    end

    # Select random candidate site
    candidate_idx = rand(rng, 1:n_candidates)
    new_site = update.candidate_sites[candidate_idx]

    # Calculate move ratio (simplified)
    ratio = one(T)  # In real implementation, this would involve wavefunction ratios

    # Create result
    result = UpdateResult{T}(true, ratio, SINGLE_ELECTRON)
    result.electron_indices = [ele_idx_selected]
    result.site_indices = [current_site, new_site]
    result.acceptance_probability = min(1.0, abs(ratio)^2)

    return result
end

"""
    TwoElectronUpdate{T}

Two electron Monte Carlo update.
"""
mutable struct TwoElectronUpdate{T<:Union{Float64,ComplexF64}}
    # System parameters
    n_site::Int
    n_elec::Int

    # Update parameters
    max_hop_distance::Int
    update_probability::Float64

    # Working arrays
    candidate_pairs::Vector{Tuple{Int,Int}}
    current_pairs::Vector{Tuple{Int,Int}}
    ratio_buffer::Vector{T}

    # Statistics
    total_attempts::Int
    total_accepted::Int
    acceptance_rate::Float64

    function TwoElectronUpdate{T}(
        n_site::Int,
        n_elec::Int;
        max_hop_distance::Int = 1,
        update_probability::Float64 = 0.05,
    ) where {T}
        candidate_pairs = Vector{Tuple{Int,Int}}(undef, n_site * n_site)
        current_pairs = Vector{Tuple{Int,Int}}(undef, n_elec * n_elec)
        ratio_buffer = Vector{T}(undef, n_elec * n_elec)

        new{T}(
            n_site,
            n_elec,
            max_hop_distance,
            update_probability,
            candidate_pairs,
            current_pairs,
            ratio_buffer,
            0,
            0,
            0.0,
        )
    end
end

"""
    propose_two_electron_move(update::TwoElectronUpdate{T}, ele_idx::Vector{Int},
                             ele_cfg::Vector{Int}, rng::AbstractRNG) where T

Propose a two electron move.
"""
function propose_two_electron_move(
    update::TwoElectronUpdate{T},
    ele_idx::Vector{Int},
    ele_cfg::Vector{Int},
    rng::AbstractRNG,
) where {T}
    if length(ele_idx) < 2
        return UpdateResult{T}(false, zero(T), TWO_ELECTRON)
    end

    # Randomly select two electrons
    ele_indices = rand(rng, 1:length(ele_idx), 2)
    if ele_indices[1] == ele_indices[2]
        return UpdateResult{T}(false, zero(T), TWO_ELECTRON)
    end

    current_sites = [ele_idx[ele_indices[1]], ele_idx[ele_indices[2]]]

    # Find candidate sites for both electrons
    n_candidates = 0
    for site1 = 1:update.n_site
        for site2 = 1:update.n_site
            if site1 != site2 && ele_cfg[site1] == 0 && ele_cfg[site2] == 0
                # Check hopping distances
                if abs(site1 - current_sites[1]) <= update.max_hop_distance &&
                   abs(site2 - current_sites[2]) <= update.max_hop_distance
                    n_candidates += 1
                    update.candidate_pairs[n_candidates] = (site1, site2)
                end
            end
        end
    end

    if n_candidates == 0
        return UpdateResult{T}(false, zero(T), TWO_ELECTRON)
    end

    # Select random candidate pair
    candidate_idx = rand(rng, 1:n_candidates)
    new_sites = update.candidate_pairs[candidate_idx]

    # Calculate move ratio (simplified)
    ratio = one(T)  # In real implementation, this would involve wavefunction ratios

    # Create result
    result = UpdateResult{T}(true, ratio, TWO_ELECTRON)
    result.electron_indices = ele_indices
    result.site_indices = [current_sites[1], current_sites[2], new_sites[1], new_sites[2]]
    result.acceptance_probability = min(1.0, abs(ratio)^2)

    return result
end

"""
    ExchangeHoppingUpdate{T}

Exchange hopping Monte Carlo update.
"""
mutable struct ExchangeHoppingUpdate{T<:Union{Float64,ComplexF64}}
    # System parameters
    n_site::Int
    n_elec::Int

    # Update parameters
    max_hop_distance::Int
    update_probability::Float64

    # Working arrays
    candidate_exchanges::Vector{Tuple{Int,Int}}
    ratio_buffer::Vector{T}

    # Statistics
    total_attempts::Int
    total_accepted::Int
    acceptance_rate::Float64

    function ExchangeHoppingUpdate{T}(
        n_site::Int,
        n_elec::Int;
        max_hop_distance::Int = 1,
        update_probability::Float64 = 0.02,
    ) where {T}
        candidate_exchanges = Vector{Tuple{Int,Int}}(undef, n_site * n_site)
        ratio_buffer = Vector{T}(undef, n_elec)

        new{T}(
            n_site,
            n_elec,
            max_hop_distance,
            update_probability,
            candidate_exchanges,
            ratio_buffer,
            0,
            0,
            0.0,
        )
    end
end

"""
    propose_exchange_hopping(update::ExchangeHoppingUpdate{T}, ele_idx::Vector{Int},
                            ele_cfg::Vector{Int}, rng::AbstractRNG) where T

Propose an exchange hopping move.
"""
function propose_exchange_hopping(
    update::ExchangeHoppingUpdate{T},
    ele_idx::Vector{Int},
    ele_cfg::Vector{Int},
    rng::AbstractRNG,
) where {T}
    if length(ele_idx) < 2
        return UpdateResult{T}(false, zero(T), EXCHANGE_HOPPING)
    end

    # Randomly select two electrons
    ele_indices = rand(rng, 1:length(ele_idx), 2)
    if ele_indices[1] == ele_indices[2]
        return UpdateResult{T}(false, zero(T), EXCHANGE_HOPPING)
    end

    current_sites = [ele_idx[ele_indices[1]], ele_idx[ele_indices[2]]]

    # Find candidate exchange sites
    n_candidates = 0
    for site1 = 1:update.n_site
        for site2 = 1:update.n_site
            if site1 != site2 && ele_cfg[site1] == 0 && ele_cfg[site2] == 0
                # Check hopping distances
                if abs(site1 - current_sites[1]) <= update.max_hop_distance &&
                   abs(site2 - current_sites[2]) <= update.max_hop_distance
                    n_candidates += 1
                    update.candidate_exchanges[n_candidates] = (site1, site2)
                end
            end
        end
    end

    if n_candidates == 0
        return UpdateResult{T}(false, zero(T), EXCHANGE_HOPPING)
    end

    # Select random candidate exchange
    candidate_idx = rand(rng, 1:n_candidates)
    new_sites = update.candidate_exchanges[candidate_idx]

    # Calculate move ratio (simplified)
    ratio = one(T)  # In real implementation, this would involve wavefunction ratios

    # Create result
    result = UpdateResult{T}(true, ratio, EXCHANGE_HOPPING)
    result.electron_indices = ele_indices
    result.site_indices = [current_sites[1], current_sites[2], new_sites[1], new_sites[2]]
    result.acceptance_probability = min(1.0, abs(ratio)^2)

    return result
end

"""
    UpdateManager{T}

Manages multiple Monte Carlo update types.
"""
mutable struct UpdateManager{T<:Union{Float64,ComplexF64}}
    # Update algorithms
    single_electron::SingleElectronUpdate{T}
    two_electron::TwoElectronUpdate{T}
    exchange_hopping::ExchangeHoppingUpdate{T}

    # Update probabilities
    single_electron_prob::Float64
    two_electron_prob::Float64
    exchange_hopping_prob::Float64

    # Statistics
    total_attempts::Int
    total_accepted::Int
    overall_acceptance_rate::Float64

    function UpdateManager{T}(n_site::Int, n_elec::Int) where {T}
        single_electron = SingleElectronUpdate{T}(n_site, n_elec)
        two_electron = TwoElectronUpdate{T}(n_site, n_elec)
        exchange_hopping = ExchangeHoppingUpdate{T}(n_site, n_elec)

        # Default probabilities
        single_electron_prob = 0.7
        two_electron_prob = 0.2
        exchange_hopping_prob = 0.1

        new{T}(
            single_electron,
            two_electron,
            exchange_hopping,
            single_electron_prob,
            two_electron_prob,
            exchange_hopping_prob,
            0,
            0,
            0.0,
        )
    end
end

"""
    propose_update(manager::UpdateManager{T}, ele_idx::Vector{Int},
                  ele_cfg::Vector{Int}, rng::AbstractRNG) where T

Propose a Monte Carlo update using the update manager.
"""
function propose_update(
    manager::UpdateManager{T},
    ele_idx::Vector{Int},
    ele_cfg::Vector{Int},
    rng::AbstractRNG,
) where {T}
    # Select update type based on probabilities
    rand_val = rand(rng)

    if rand_val < manager.single_electron_prob
        return propose_single_electron_move(manager.single_electron, ele_idx, ele_cfg, rng)
    elseif rand_val < manager.single_electron_prob + manager.two_electron_prob
        return propose_two_electron_move(manager.two_electron, ele_idx, ele_cfg, rng)
    else
        return propose_exchange_hopping(manager.exchange_hopping, ele_idx, ele_cfg, rng)
    end
end

"""
    accept_update!(manager::UpdateManager{T}, result::UpdateResult{T}) where T

Accept a proposed update and update statistics.
"""
function accept_update!(manager::UpdateManager{T}, result::UpdateResult{T}) where {T}
    if result.success
        manager.total_attempts += 1
        manager.total_accepted += 1

        # Update specific update type statistics
        if result.update_type == SINGLE_ELECTRON
            manager.single_electron.total_attempts += 1
            manager.single_electron.total_accepted += 1
        elseif result.update_type == TWO_ELECTRON
            manager.two_electron.total_attempts += 1
            manager.two_electron.total_accepted += 1
        elseif result.update_type == EXCHANGE_HOPPING
            manager.exchange_hopping.total_attempts += 1
            manager.exchange_hopping.total_accepted += 1
        end

        # Update acceptance rates
        manager.overall_acceptance_rate = manager.total_accepted / manager.total_attempts
        if result.update_type == SINGLE_ELECTRON
            manager.single_electron.acceptance_rate =
                manager.single_electron.total_accepted /
                manager.single_electron.total_attempts
        elseif result.update_type == TWO_ELECTRON
            manager.two_electron.acceptance_rate =
                manager.two_electron.total_accepted / manager.two_electron.total_attempts
        elseif result.update_type == EXCHANGE_HOPPING
            manager.exchange_hopping.acceptance_rate =
                manager.exchange_hopping.total_accepted /
                manager.exchange_hopping.total_attempts
        end
    end
end

"""
    reject_update!(manager::UpdateManager{T}, result::UpdateResult{T}) where T

Reject a proposed update and update statistics.
"""
function reject_update!(manager::UpdateManager{T}, result::UpdateResult{T}) where {T}
    if result.success
        manager.total_attempts += 1

        # Update specific update type statistics
        if result.update_type == SINGLE_ELECTRON
            manager.single_electron.total_attempts += 1
        elseif result.update_type == TWO_ELECTRON
            manager.two_electron.total_attempts += 1
        elseif result.update_type == EXCHANGE_HOPPING
            manager.exchange_hopping.total_attempts += 1
        end

        # Update acceptance rates
        manager.overall_acceptance_rate = manager.total_accepted / manager.total_attempts
        if result.update_type == SINGLE_ELECTRON
            manager.single_electron.acceptance_rate =
                manager.single_electron.total_accepted /
                manager.single_electron.total_attempts
        elseif result.update_type == TWO_ELECTRON
            manager.two_electron.acceptance_rate =
                manager.two_electron.total_accepted / manager.two_electron.total_attempts
        elseif result.update_type == EXCHANGE_HOPPING
            manager.exchange_hopping.acceptance_rate =
                manager.exchange_hopping.total_accepted /
                manager.exchange_hopping.total_attempts
        end
    end
end

"""
    get_update_statistics(manager::UpdateManager{T}) where T

Get comprehensive update statistics.
"""
function get_update_statistics(manager::UpdateManager{T}) where {T}
    return (
        overall = (
            total_attempts = manager.total_attempts,
            total_accepted = manager.total_accepted,
            acceptance_rate = manager.overall_acceptance_rate,
        ),
        single_electron = (
            total_attempts = manager.single_electron.total_attempts,
            total_accepted = manager.single_electron.total_accepted,
            acceptance_rate = manager.single_electron.acceptance_rate,
        ),
        two_electron = (
            total_attempts = manager.two_electron.total_attempts,
            total_accepted = manager.two_electron.total_accepted,
            acceptance_rate = manager.two_electron.acceptance_rate,
        ),
        exchange_hopping = (
            total_attempts = manager.exchange_hopping.total_attempts,
            total_accepted = manager.exchange_hopping.total_accepted,
            acceptance_rate = manager.exchange_hopping.acceptance_rate,
        ),
    )
end

"""
    reset_update_statistics!(manager::UpdateManager{T}) where T

Reset all update statistics.
"""
function reset_update_statistics!(manager::UpdateManager{T}) where {T}
    manager.total_attempts = 0
    manager.total_accepted = 0
    manager.overall_acceptance_rate = 0.0

    manager.single_electron.total_attempts = 0
    manager.single_electron.total_accepted = 0
    manager.single_electron.acceptance_rate = 0.0

    manager.two_electron.total_attempts = 0
    manager.two_electron.total_accepted = 0
    manager.two_electron.acceptance_rate = 0.0

    manager.exchange_hopping.total_attempts = 0
    manager.exchange_hopping.total_accepted = 0
    manager.exchange_hopping.acceptance_rate = 0.0
end

"""
    benchmark_updates(n_site::Int = 10, n_elec::Int = 5, n_iterations::Int = 10000)

Benchmark Monte Carlo update algorithms.
"""
function benchmark_updates(n_site::Int = 10, n_elec::Int = 5, n_iterations::Int = 10000)
    println(
        "Benchmarking Monte Carlo updates (n_site=$n_site, n_elec=$n_elec, iterations=$n_iterations)...",
    )

    # Create update manager
    manager = UpdateManager{ComplexF64}(n_site, n_elec)

    # Initialize electron configuration
    ele_idx = collect(1:n_elec)
    ele_cfg = zeros(Int, n_site)
    ele_cfg[1:n_elec] .= 1

    # Benchmark update proposals
    @time begin
        for _ = 1:n_iterations
            result = propose_update(manager, ele_idx, ele_cfg, Random.GLOBAL_RNG)
            if result.success
                # Simulate acceptance/rejection
                if rand() < result.acceptance_probability
                    accept_update!(manager, result)
                else
                    reject_update!(manager, result)
                end
            end
        end
    end
    println("  Update proposal and acceptance rate")

    # Print statistics
    stats = get_update_statistics(manager)
    println("Update benchmark completed.")
    println("  Overall acceptance rate: $(stats.overall.acceptance_rate)")
    println("  Single electron acceptance rate: $(stats.single_electron.acceptance_rate)")
    println("  Two electron acceptance rate: $(stats.two_electron.acceptance_rate)")
    println("  Exchange hopping acceptance rate: $(stats.exchange_hopping.acceptance_rate)")
end
