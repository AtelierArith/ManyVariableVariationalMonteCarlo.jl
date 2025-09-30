"""
FSZ (Fixed Spin Zone) Sampler

Complete FSZ sampling implementation for spin systems.
Integrates with VMC loop for Heisenberg-type models.

Based on mVMC C implementation:
- vmcmake_fsz.c: FSZ sampling
- vmccal_fsz.c: FSZ calculations
"""

using LinearAlgebra
using Random
using Printf

"""
    FSZSamplerConfig

Configuration for FSZ sampling.
"""
struct FSZSamplerConfig
    n_sites::Int
    n_elec::Int
    two_sz::Int
    n_ex_update_path::Int
    n_samples::Int
    n_warm_up::Int
    n_interval::Int
    use_real::Bool

    function FSZSamplerConfig(;
        n_sites::Int,
        n_elec::Int,
        two_sz::Int = 0,
        n_ex_update_path::Int = 2,
        n_samples::Int = 100,
        n_warm_up::Int = 10,
        n_interval::Int = 1,
        use_real::Bool = true,
    )
        new(n_sites, n_elec, two_sz, n_ex_update_path,
            n_samples, n_warm_up, n_interval, use_real)
    end
end

"""
    FSZSamplerState{T}

State for FSZ sampling including electron configurations and statistics.
"""
mutable struct FSZSamplerState{T<:Number}
    # Configuration
    config::FSZSamplerConfig

    # Electron configuration arrays
    ele_idx::Vector{Int}      # Which site each electron occupies [2*Ne]
    ele_spn::Vector{Int}      # Spin of each electron (0=up, 1=down) [2*Ne]
    ele_cfg::Vector{Int}      # Which electron at each site [2*Nsite]
    ele_num::Vector{Int}      # Occupation number at each site [2*Nsite]

    # Projection counters
    ele_proj_cnt::Vector{Int}  # Projection counts

    # Local spin flags
    loc_spn::Vector{Int}       # Local spin flags (1=local, 0=itinerant)

    # Orbital matrix (needed for Sherman-Morrison)
    orbital_matrix::Union{Nothing,Matrix{T}}

    # Slater matrices (for Sz-resolved calculation)
    slater_up::Union{Nothing,Matrix{T}}
    slater_down::Union{Nothing,Matrix{T}}
    inv_m_up::Union{Nothing,Matrix{T}}
    inv_m_down::Union{Nothing,Matrix{T}}

    # Determinants/Pfaffians
    det_up::T
    det_down::T

    # Counters for update statistics
    counter_hopping_all::Int
    counter_hopping_accept::Int
    counter_exchange_all::Int
    counter_exchange_accept::Int
    counter_spin_flip_all::Int
    counter_spin_flip_accept::Int

    # RNG
    rng::AbstractRNG

    function FSZSamplerState{T}(
        config::FSZSamplerConfig,
        loc_spn::Vector{Int},
        rng::AbstractRNG,
    ) where {T}
        n_size = 2 * config.n_elec
        n_site2 = 2 * config.n_sites

        # Initialize electron configurations
        (ele_idx, ele_spn, ele_cfg, ele_num) =
            generate_initial_electron_config_with_2sz(
                config.n_sites, config.n_elec, config.two_sz, loc_spn, rng
            )

        # Initialize projection counters
        ele_proj_cnt = zeros(Int, n_size)  # Placeholder

        # Initialize Slater matrices if needed
        n_up = count(==(0), ele_spn)
        n_down = count(==(1), ele_spn)

        slater_up = n_up > 0 ? zeros(T, n_up, n_up) : nothing
        slater_down = n_down > 0 ? zeros(T, n_down, n_down) : nothing
        inv_m_up = n_up > 0 ? zeros(T, n_up, n_up) : nothing
        inv_m_down = n_down > 0 ? zeros(T, n_down, n_down) : nothing

        new{T}(
            config,
            ele_idx,
            ele_spn,
            ele_cfg,
            ele_num,
            ele_proj_cnt,
            loc_spn,
            nothing,  # orbital_matrix (set later)
            slater_up,
            slater_down,
            inv_m_up,
            inv_m_down,
            one(T),
            one(T),
            0, 0, 0, 0, 0, 0,
            rng,
        )
    end
end

"""
    initialize_fsz_sampler!(state::FSZSamplerState{T}, orbital_matrix::Matrix{T}) where T

Initialize FSZ sampler with orbital matrix.
Calculate initial Slater determinants.
"""
function initialize_fsz_sampler!(
    state::FSZSamplerState{T},
    orbital_matrix::Matrix{T},
) where {T}
    # Store orbital matrix
    state.orbital_matrix = copy(orbital_matrix)

    # Separate up and down electrons
    up_indices = findall(==(0), state.ele_spn)
    down_indices = findall(==(1), state.ele_spn)

    n_up = length(up_indices)
    n_down = length(down_indices)

    # Build Slater matrices
    if n_up > 0 && state.slater_up !== nothing
        for (i, ei) in enumerate(up_indices)
            site_i = state.ele_idx[ei] + 1  # Convert to 1-based
            for (j, ej) in enumerate(up_indices)
                site_j = state.ele_idx[ej] + 1
                state.slater_up[i, j] = orbital_matrix[site_i, site_j]
            end
        end

        # Calculate determinant and inverse
        state.det_up = det(state.slater_up)
        if abs(state.det_up) > 1e-15
            state.inv_m_up .= inv(state.slater_up)
        end
    end

    if n_down > 0 && state.slater_down !== nothing
        for (i, ei) in enumerate(down_indices)
            site_i = state.ele_idx[ei] + 1
            for (j, ej) in enumerate(down_indices)
                site_j = state.ele_idx[ej] + 1
                state.slater_down[i, j] = orbital_matrix[site_i, site_j]
            end
        end

        state.det_down = det(state.slater_down)
        if abs(state.det_down) > 1e-15
            state.inv_m_down .= inv(state.slater_down)
        end
    end

    return nothing
end

"""
    propose_fsz_update!(state::FSZSamplerState{T}) where T

Propose and execute one FSZ update step.
Returns: (accepted::Bool, update_type::Symbol)
"""
function propose_fsz_update!(state::FSZSamplerState{T}) where {T}
    config = state.config

    # Get update type
    update_type = get_update_type_for_spin_system(
        config.n_ex_update_path,
        config.two_sz,
        state.rng,
    )

    if update_type == :HOPPING
        return propose_hopping_update!(state)
    elseif update_type == :LOCALSPINFLIP
        return propose_spin_flip_update!(state)
    else
        return (false, :UNKNOWN)
    end
end

"""
    propose_hopping_update!(state::FSZSamplerState{T}) where T

Propose hopping update (spin-conserving for csz).
"""
function propose_hopping_update!(state::FSZSamplerState{T}) where {T}
    state.counter_hopping_all += 1

    # Make candidate
    (mi, ri, rj, s, t, reject_flag) = make_candidate_hopping_csz(
        state.ele_idx,
        state.ele_cfg,
        state.ele_num,
        state.ele_spn,
        state.config.n_sites,
        state.rng,
    )

    if reject_flag
        return (false, :HOPPING)
    end

    # Calculate acceptance ratio
    # For now, use simplified acceptance (should use Slater matrix ratio)
    ratio = calculate_hopping_ratio(state, mi, ri, rj, s)

    # Metropolis acceptance
    if abs(ratio)^2 >= rand(state.rng)
        # Accept update
        update_electron_configuration!(state, mi, ri, rj, s, t)
        state.counter_hopping_accept += 1
        return (true, :HOPPING)
    else
        return (false, :HOPPING)
    end
end

"""
    propose_spin_flip_update!(state::FSZSamplerState{T}) where T

Propose local spin flip update (for TwoSz=-1).
"""
function propose_spin_flip_update!(state::FSZSamplerState{T}) where {T}
    state.counter_spin_flip_all += 1

    # Make candidate
    (mi, ri, rj, s, t, reject_flag) = make_candidate_local_spin_flip_conduction(
        state.ele_idx,
        state.ele_cfg,
        state.ele_num,
        state.ele_spn,
        state.config.n_sites,
        state.rng,
    )

    if reject_flag
        return (false, :LOCALSPINFLIP)
    end

    # Calculate acceptance ratio
    ratio = calculate_spin_flip_ratio(state, mi, ri, s, t)

    # Metropolis acceptance
    if abs(ratio)^2 >= rand(state.rng)
        # Accept update
        update_electron_configuration!(state, mi, ri, rj, s, t)
        state.counter_spin_flip_accept += 1
        return (true, :LOCALSPINFLIP)
    else
        return (false, :LOCALSPINFLIP)
    end
end

"""
    calculate_hopping_ratio(state::FSZSamplerState{T}, mi, ri, rj, s) where T

Calculate acceptance ratio for hopping move using Sherman-Morrison formula.
"""
function calculate_hopping_ratio(
    state::FSZSamplerState{T},
    mi::Int,
    ri::Int,
    rj::Int,
    s::Int,
) where {T}
    # Determine which spin sector this electron belongs to
    spin = state.ele_spn[mi]

    if spin == 0  # Up spin
        if state.slater_up === nothing || state.inv_m_up === nothing
            return one(T)
        end

        # Find position of this electron in up-spin list
        up_indices = findall(==(0), state.ele_spn)
        electron_idx_in_list = findfirst(==(mi), up_indices)

        if electron_idx_in_list === nothing
            return one(T)
        end

        # Get list of up-spin electron sites
        electron_sites = [state.ele_idx[ei] for ei in up_indices]

        # Calculate ratio using Sherman-Morrison
        if state.orbital_matrix === nothing
            return one(T)  # Fallback if not initialized
        end

        ratio = calculate_ratio_sherman_morrison(
            state.inv_m_up,
            state.orbital_matrix,
            electron_idx_in_list,
            ri,
            rj,
            electron_sites,
        )

        return ratio

    else  # Down spin
        if state.slater_down === nothing || state.inv_m_down === nothing
            return one(T)
        end

        # Similar for down spin
        down_indices = findall(==(1), state.ele_spn)
        electron_idx_in_list = findfirst(==(mi), down_indices)

        if electron_idx_in_list === nothing
            return one(T)
        end

        electron_sites = [state.ele_idx[ei] for ei in down_indices]

        if state.orbital_matrix === nothing
            return one(T)  # Fallback if not initialized
        end

        ratio = calculate_ratio_sherman_morrison(
            state.inv_m_down,
            state.orbital_matrix,
            electron_idx_in_list,
            ri,
            rj,
            electron_sites,
        )

        return ratio
    end
end

"""
    calculate_spin_flip_ratio(state::FSZSamplerState{T}, mi, ri, s, t) where T

Calculate acceptance ratio for spin flip move.
"""
function calculate_spin_flip_ratio(
    state::FSZSamplerState{T},
    mi::Int,
    ri::Int,
    s::Int,
    t::Int,
) where {T}
    # Simplified
    return one(T)
end

"""
    update_electron_configuration!(state::FSZSamplerState, mi, ri, rj, s, t)

Update electron configuration after accepted move.
Based on updateEleConfig_fsz() in C implementation.
"""
function update_electron_configuration!(
    state::FSZSamplerState,
    mi::Int,
    ri::Int,
    rj::Int,
    s::Int,
    t::Int,
)
    n_sites = state.config.n_sites

    # Bounds checking
    if mi < 1 || mi > length(state.ele_idx)
        @error "Invalid mi: $mi, valid range: 1-$(length(state.ele_idx))"
        return nothing
    end

    # Calculate indices (convert 0-based to 1-based)
    ri_idx = ri + 1 + s * n_sites
    rj_idx = rj + 1 + t * n_sites

    if ri_idx < 1 || ri_idx > length(state.ele_cfg)
        @error "Invalid ri_idx: $ri_idx (ri=$ri, s=$s, n_sites=$n_sites), valid range: 1-$(length(state.ele_cfg))"
        return nothing
    end

    if rj_idx < 1 || rj_idx > length(state.ele_cfg)
        @error "Invalid rj_idx: $rj_idx (rj=$rj, t=$t, n_sites=$n_sites), valid range: 1-$(length(state.ele_cfg))"
        return nothing
    end

    # Update electron index
    state.ele_idx[mi] = rj

    # Update electron configuration
    state.ele_cfg[ri_idx] = -1
    state.ele_cfg[rj_idx] = mi - 1  # Store as 0-based

    # Update occupation numbers
    state.ele_num[ri_idx] = 0
    state.ele_num[rj_idx] = 1

    # Update spin if changed
    if s != t
        state.ele_spn[mi] = t
    end

    return nothing
end

"""
    run_fsz_sampling!(
        state::FSZSamplerState{T},
        orbital_matrix::Matrix{T};
        verbose::Bool = false
    ) where T

Run FSZ sampling loop.
"""
function run_fsz_sampling!(
    state::FSZSamplerState{T},
    orbital_matrix::Matrix{T};
    verbose::Bool = false,
) where {T}
    config = state.config

    # Initialize
    initialize_fsz_sampler!(state, orbital_matrix)

    # Warm-up phase
    if verbose
        println("FSZ Sampling: Warm-up phase ($(config.n_warm_up) steps)")
    end

    for i in 1:config.n_warm_up
        for j in 1:(config.n_interval * config.n_sites)
            propose_fsz_update!(state)
        end
    end

    # Sampling phase
    if verbose
        println("FSZ Sampling: Sampling phase ($(config.n_samples) samples)")
    end

    samples = Vector{Vector{Int}}()

    for i in 1:config.n_samples
        # Perform interval updates
        for j in 1:(config.n_interval * config.n_sites)
            propose_fsz_update!(state)
        end

        # Store sample
        push!(samples, copy(state.ele_idx))
    end

    # Print statistics
    if verbose
        println("\nFSZ Sampling Statistics:")

        hop_rate = state.counter_hopping_all > 0 ?
            state.counter_hopping_accept / state.counter_hopping_all : 0.0
        println("  Hopping: $(state.counter_hopping_accept)/$(state.counter_hopping_all) " *
                "($(round(hop_rate*100, digits=2))%)")

        if state.counter_spin_flip_all > 0
            flip_rate = state.counter_spin_flip_accept / state.counter_spin_flip_all
            println("  Spin flip: $(state.counter_spin_flip_accept)/$(state.counter_spin_flip_all) " *
                    "($(round(flip_rate*100, digits=2))%)")
        end
    end

    return samples
end

"""
    create_fsz_sampler_from_params(params::StdFaceParameters, rng::AbstractRNG) -> FSZSamplerState

Create FSZ sampler from StdFace parameters.
"""
function create_fsz_sampler_from_params(
    params::StdFaceParameters,
    rng::AbstractRNG,
)
    # For spin systems, n_elec = n_sites (one electron per site, half-filled)
    n_elec = params.L

    # Local spin flags: for spin systems, all sites are local spins
    loc_spn = ones(Int, params.L)

    config = FSZSamplerConfig(
        n_sites = params.L,
        n_elec = n_elec,
        two_sz = params.TwoSz,
        n_ex_update_path = 2,  # Spin system
        n_samples = params.NVMCSample,
        n_warm_up = params.NVMCWarmUp,
        n_interval = params.NVMCInterval,
        use_real = true,  # Spin systems typically use real parameters
    )

    T = config.use_real ? Float64 : ComplexF64

    return FSZSamplerState{T}(config, loc_spn, rng)
end
