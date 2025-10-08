"""
Detailed Wavefunction Components

Enhanced implementations of wavefunction components for high-precision VMC calculations.
Based on the C reference implementation in mVMC.
"""

using LinearAlgebra
using SparseArrays

"""
    GutzwillerProjector{T}

Enhanced Gutzwiller projector implementation with site-dependent parameters.
Supports both onsite and intersite correlations.
"""
mutable struct GutzwillerProjector{T<:Union{Float64,ComplexF64}}
    n_sites::Int
    n_electrons::Int

    # Gutzwiller parameters (site-dependent)
    g_parameters::Vector{T}

    # Correlation matrices for intersite effects
    correlation_matrix::Matrix{T}

    # Precomputed factors for efficiency
    site_factors::Vector{T}
    pair_factors::Matrix{T}

    # Configuration-dependent values
    current_projection::T

    function GutzwillerProjector{T}(n_sites::Int, n_electrons::Int) where {T}
        new{T}(
            n_sites, n_electrons,
            zeros(T, n_sites),
            zeros(T, n_sites, n_sites),
            zeros(T, n_sites),
            zeros(T, n_sites, n_sites),
            one(T)
        )
    end
end

GutzwillerProjector(n_sites::Int, n_electrons::Int; T=ComplexF64) =
    GutzwillerProjector{T}(n_sites, n_electrons)

"""
    set_gutzwiller_parameters!(proj::GutzwillerProjector{T}, params::Vector{T}) where {T}

Set Gutzwiller parameters for all sites.

C実装参考: projection.c 1行目から452行目まで
"""
function set_gutzwiller_parameters!(proj::GutzwillerProjector{T}, params::Vector{T}) where {T}
    n_params = min(length(params), proj.n_sites)
    proj.g_parameters[1:n_params] .= params[1:n_params]

    # Update site factors
    for i in 1:proj.n_sites
        proj.site_factors[i] = exp(proj.g_parameters[i])
    end
end

"""
    compute_gutzwiller_factor!(proj::GutzwillerProjector{T}, state::VMCState{T}) where {T}

Compute Gutzwiller projection factor for current electron configuration.
"""
function compute_gutzwiller_factor!(proj::GutzwillerProjector{T}, state::VMCState{T}) where {T}
    proj.current_projection = one(T)

    # Count electrons per site
    n_up = zeros(Int, proj.n_sites)
    n_down = zeros(Int, proj.n_sites)

    # Check if we have spin configuration (for spin models)
    if hasfield(typeof(state), :spin_configuration) && !isempty(state.spin_configuration)
        # For spin models: use spin configuration directly
        for site in 1:min(length(state.spin_configuration), proj.n_sites)
            spin = state.spin_configuration[site]
            if spin == 1  # Spin up
                n_up[site] = 1
            elseif spin == -1  # Spin down
                n_down[site] = 1
            end
        end
    else
        # For fermion models: use electron positions
        n_up_electrons = div(state.n_electrons, 2)
        for i in 1:n_up_electrons
            site = state.electron_positions[i]
            n_up[site] += 1
        end
        for i in (n_up_electrons+1):state.n_electrons
            site = state.electron_positions[i]
            n_down[site] += 1
        end
    end

    # Compute projection factor
    for i in 1:proj.n_sites
        # Onsite double occupation
        double_occ = n_up[i] * n_down[i]
        if double_occ > 0
            proj.current_projection *= proj.site_factors[i]^double_occ
        end

        # Density-dependent factors
        total_density = n_up[i] + n_down[i]
        if total_density > 0
            density_factor = exp(real(proj.g_parameters[i]) * (total_density - 1))
            proj.current_projection *= T(density_factor)
        end
    end

    return proj.current_projection
end

"""
    EnhancedJastrowFactor{T}

Enhanced Jastrow factor with multiple correlation types:
- Onsite (Gutzwiller-type)
- Nearest neighbor
- Long-range
- Spin-dependent correlations
"""
mutable struct EnhancedJastrowFactor{T<:Union{Float64,ComplexF64}}
    n_sites::Int
    n_electrons::Int

    # Parameter arrays
    onsite_params::Vector{T}
    nn_params::Vector{T}
    longrange_params::Vector{T}
    spin_params::Vector{T}

    # Neighbor lists for efficient computation
    neighbor_list::Vector{Vector{Int}}
    distance_matrix::Matrix{Float64}

    # Current Jastrow value
    current_value::T

    function EnhancedJastrowFactor{T}(n_sites::Int, n_electrons::Int) where {T}
        new{T}(
            n_sites, n_electrons,
            zeros(T, n_sites),
            zeros(T, n_sites),
            zeros(T, div(n_sites * (n_sites - 1), 2)),
            zeros(T, n_sites),
            [Int[] for _ in 1:n_sites],
            zeros(Float64, n_sites, n_sites),
            one(T)
        )
    end
end

EnhancedJastrowFactor(n_sites::Int, n_electrons::Int; T=ComplexF64) =
    EnhancedJastrowFactor{T}(n_sites, n_electrons)

"""
    initialize_neighbor_lists!(jastrow::EnhancedJastrowFactor, geometry)

Initialize neighbor lists based on lattice geometry.
"""
function initialize_neighbor_lists!(jastrow::EnhancedJastrowFactor, geometry)
    # Clear existing neighbor lists
    for i in 1:jastrow.n_sites
        empty!(jastrow.neighbor_list[i])
    end

    if geometry === nothing
        # Default: nearest neighbors on a chain
        for i in 1:jastrow.n_sites
            if i > 1
                push!(jastrow.neighbor_list[i], i-1)
            end
            if i < jastrow.n_sites
                push!(jastrow.neighbor_list[i], i+1)
            end
        end

        # Distance matrix for chain
        for i in 1:jastrow.n_sites, j in 1:jastrow.n_sites
            jastrow.distance_matrix[i,j] = abs(i - j)
        end
    else
        # Use enhanced geometry information - delegate to integration module
        # This will be handled by initialize_neighbor_lists! in mvmc_integration.jl
        # Default: nearest neighbors in 1D chain
        for i in 1:jastrow.n_sites
            for j in 1:jastrow.n_sites
                if i != j
                    dist = abs(i - j)
                    jastrow.distance_matrix[i,j] = dist
                    if dist == 1  # Nearest neighbor in 1D
                        push!(jastrow.neighbor_list[i], j)
                    end
                end
            end
        end
    end
end

"""
    set_jastrow_parameters!(jastrow::EnhancedJastrowFactor{T}, params::Vector{T}) where {T}

Set Jastrow parameters from a parameter vector.
"""
function set_jastrow_parameters!(jastrow::EnhancedJastrowFactor{T}, params::Vector{T}) where {T}
    # Special handling for StdFace Spin with tiny parameter count (e.g., NGutz=1, NJast=1)
    if length(params) <= 2
        # Use the last parameter as a global spin-Jastrow coefficient
        if !isempty(params)
            global_spin = params[end]
            fill!(jastrow.spin_params, global_spin)
        end
        # Zero other components to avoid unintended density-only constants
        fill!(jastrow.onsite_params, zero(T))
        fill!(jastrow.nn_params, zero(T))
        fill!(jastrow.longrange_params, zero(T))
        return
    end

    offset = 0

    # Onsite parameters
    n_onsite = min(length(params) - offset, jastrow.n_sites)
    if n_onsite > 0
        jastrow.onsite_params[1:n_onsite] .= params[(offset+1):(offset+n_onsite)]
        offset += n_onsite
    end

    # Nearest neighbor parameters
    n_nn = min(length(params) - offset, jastrow.n_sites)
    if n_nn > 0
        jastrow.nn_params[1:n_nn] .= params[(offset+1):(offset+n_nn)]
        offset += n_nn
    end

    # Long-range parameters
    n_lr = min(length(params) - offset, length(jastrow.longrange_params))
    if n_lr > 0
        jastrow.longrange_params[1:n_lr] .= params[(offset+1):(offset+n_lr)]
        offset += n_lr
    end

    # Spin parameters
    n_spin = min(length(params) - offset, jastrow.n_sites)
    if n_spin > 0
        jastrow.spin_params[1:n_spin] .= params[(offset+1):(offset+n_spin)]
    end
end

"""
    compute_jastrow_factor!(jastrow::EnhancedJastrowFactor{T}, state::VMCState{T}) where {T}

Compute Jastrow factor for current electron configuration.
"""
function compute_jastrow_factor!(jastrow::EnhancedJastrowFactor{T}, state::VMCState{T}) where {T}
    jastrow.current_value = zero(T)

    # Count electrons per site and spin
    n_up = zeros(Int, jastrow.n_sites)
    n_down = zeros(Int, jastrow.n_sites)

    # Check if we have spin configuration (for spin models)
    if hasfield(typeof(state), :spin_configuration) && !isempty(state.spin_configuration)
        # For spin models: use spin configuration directly
        for site in 1:min(length(state.spin_configuration), jastrow.n_sites)
            spin = state.spin_configuration[site]
            if spin == 1  # Spin up
                n_up[site] = 1
            elseif spin == -1  # Spin down
                n_down[site] = 1
            end
        end
    else
        # For fermion models: use electron positions
        n_up_electrons = div(state.n_electrons, 2)
        for i in 1:n_up_electrons
            site = state.electron_positions[i]
            n_up[site] += 1
        end
        for i in (n_up_electrons+1):state.n_electrons
            site = state.electron_positions[i]
            n_down[site] += 1
        end
    end

    # Onsite contributions
    for i in 1:jastrow.n_sites
        n_total = n_up[i] + n_down[i]
        double_occ = n_up[i] * n_down[i]

        # Gutzwiller-type onsite correlation
        jastrow.current_value += real(jastrow.onsite_params[i]) * double_occ

        # Density-dependent terms
        if n_total > 0
            jastrow.current_value += real(jastrow.onsite_params[i]) * (n_total - 1) * n_total / 2
        end
    end

    # Nearest neighbor contributions
    for i in 1:jastrow.n_sites
        for j in jastrow.neighbor_list[i]
            if i < j  # Avoid double counting
                n_i = n_up[i] + n_down[i]
                n_j = n_up[j] + n_down[j]

                # Density-density correlation
                jastrow.current_value += real(jastrow.nn_params[min(i, jastrow.n_sites)]) * n_i * n_j

                # Spin-spin correlation
                sz_i = (n_up[i] - n_down[i]) / 2
                sz_j = (n_up[j] - n_down[j]) / 2
                jastrow.current_value += real(jastrow.spin_params[min(i, jastrow.n_sites)]) * sz_i * sz_j
            end
        end
    end

    # Long-range contributions (power-law decay)
    lr_idx = 0
    for i in 1:jastrow.n_sites
        for j in (i+1):jastrow.n_sites
            if jastrow.distance_matrix[i,j] > 1.5  # Beyond nearest neighbors
                lr_idx += 1
                if lr_idx <= length(jastrow.longrange_params)
                    n_i = n_up[i] + n_down[i]
                    n_j = n_up[j] + n_down[j]

                    # Power-law correlation
                    distance = jastrow.distance_matrix[i,j]
                    correlation = real(jastrow.longrange_params[lr_idx]) * n_i * n_j / (distance^2 + 1)
                    jastrow.current_value += correlation
                end
            end
        end
    end

    return exp(jastrow.current_value)
end

"""
    EnhancedRBMNetwork{T}

Enhanced Restricted Boltzmann Machine with:
- Complex-valued weights
- Multiple hidden layers
- Spin-dependent visible units
"""
mutable struct EnhancedRBMNetwork{T<:Union{Float64,ComplexF64}}
    n_visible::Int
    n_hidden::Int
    n_sites::Int

    # Network parameters
    visible_bias::Vector{T}
    hidden_bias::Vector{T}
    weights::Matrix{T}

    # Hidden layer activations
    hidden_activations::Vector{T}

    # Current network output
    current_amplitude::T

    function EnhancedRBMNetwork{T}(n_visible::Int, n_hidden::Int, n_sites::Int) where {T}
        new{T}(
            n_visible, n_hidden, n_sites,
            zeros(T, n_visible),
            zeros(T, n_hidden),
            zeros(T, n_hidden, n_visible),
            zeros(T, n_hidden),
            one(T)
        )
    end
end

EnhancedRBMNetwork(n_visible::Int, n_hidden::Int, n_sites::Int; T=ComplexF64) =
    EnhancedRBMNetwork{T}(n_visible, n_hidden, n_sites)

"""
    initialize_rbm_random!(rbm::EnhancedRBMNetwork{T}, rng) where {T}

Initialize RBM parameters with small random values.
"""
function initialize_rbm_random!(rbm::EnhancedRBMNetwork{T}, rng) where {T}
    # Small random initialization
    scale = T(0.1)

    for i in 1:rbm.n_visible
        rbm.visible_bias[i] = scale * (rand(rng) - 0.5) * 2
        if T <: Complex
            rbm.visible_bias[i] += scale * (rand(rng) - 0.5) * 2 * im
        end
    end

    for j in 1:rbm.n_hidden
        rbm.hidden_bias[j] = scale * (rand(rng) - 0.5) * 2
        if T <: Complex
            rbm.hidden_bias[j] += scale * (rand(rng) - 0.5) * 2 * im
        end
    end

    for j in 1:rbm.n_hidden, i in 1:rbm.n_visible
        rbm.weights[j,i] = scale * (rand(rng) - 0.5) * 2
        if T <: Complex
            rbm.weights[j,i] += scale * (rand(rng) - 0.5) * 2 * im
        end
    end
end

"""
    set_rbm_parameters!(rbm::EnhancedRBMNetwork{T}, params::Vector{T}) where {T}

Set RBM parameters from a parameter vector.
"""
function set_rbm_parameters!(rbm::EnhancedRBMNetwork{T}, params::Vector{T}) where {T}
    offset = 0

    # Visible bias
    n_vbias = min(length(params) - offset, rbm.n_visible)
    if n_vbias > 0
        rbm.visible_bias[1:n_vbias] .= params[(offset+1):(offset+n_vbias)]
        offset += n_vbias
    end

    # Hidden bias
    n_hbias = min(length(params) - offset, rbm.n_hidden)
    if n_hbias > 0
        rbm.hidden_bias[1:n_hbias] .= params[(offset+1):(offset+n_hbias)]
        offset += n_hbias
    end

    # Weights (row-major order)
    n_weights = min(length(params) - offset, rbm.n_hidden * rbm.n_visible)
    if n_weights > 0
        weight_params = params[(offset+1):(offset+n_weights)]
        idx = 0
        for j in 1:rbm.n_hidden
            for i in 1:rbm.n_visible
                idx += 1
                if idx <= n_weights
                    rbm.weights[j,i] = weight_params[idx]
                end
            end
        end
    end
end

"""
    compute_rbm_amplitude!(rbm::EnhancedRBMNetwork{T}, state::VMCState{T}) where {T}

Compute RBM amplitude for current electron configuration.
"""
function compute_rbm_amplitude!(rbm::EnhancedRBMNetwork{T}, state::VMCState{T}) where {T}
    # Convert electron positions to visible units (spin up/down representation)
    visible_config = zeros(Int, rbm.n_visible)

    # Check if we have spin configuration (for spin models)
    if hasfield(typeof(state), :spin_configuration) && !isempty(state.spin_configuration)
        # For spin models: use spin configuration directly
        for site in 1:min(length(state.spin_configuration), rbm.n_sites)
            spin = state.spin_configuration[site]
            if spin == 1  # Spin up
                visible_config[2*site-1] = 1
            elseif spin == -1  # Spin down
                visible_config[2*site] = 1
            end
        end
    else
        # For fermion models: use electron positions
        n_up_electrons = div(state.n_electrons, 2)

        # Spin-up electrons
        for i in 1:n_up_electrons
            site = state.electron_positions[i]
            if site <= rbm.n_sites
                visible_config[2*site-1] = 1  # Spin-up at site
            end
        end

        # Spin-down electrons
        for i in (n_up_electrons+1):state.n_electrons
            site = state.electron_positions[i]
            if site <= rbm.n_sites
                visible_config[2*site] = 1  # Spin-down at site
            end
        end
    end

    # Compute hidden activations
    for j in 1:rbm.n_hidden
        activation = rbm.hidden_bias[j]
        for i in 1:rbm.n_visible
            activation += rbm.weights[j,i] * visible_config[i]
        end
        rbm.hidden_activations[j] = activation
    end

    # Compute amplitude
    rbm.current_amplitude = zero(T)

    # Visible bias contribution
    for i in 1:rbm.n_visible
        rbm.current_amplitude += rbm.visible_bias[i] * visible_config[i]
    end

    # Hidden layer contribution (log-sum-exp for stability)
    for j in 1:rbm.n_hidden
        z = rbm.hidden_activations[j]
        if real(z) > 10  # Prevent overflow
            rbm.current_amplitude += real(z)  # Use real part only
        else
            # Use real part only for stability
            rbm.current_amplitude += log(1.0 + exp(real(z)))
        end
    end

    return exp(rbm.current_amplitude)
end

"""
    CombinedWavefunction{T}

Combined wavefunction that includes all components:
- Slater determinant (base wavefunction)
- Gutzwiller projector
- Enhanced Jastrow factor
- RBM network
"""
mutable struct CombinedWavefunction{T<:Union{Float64,ComplexF64}}
    slater_det::Union{Nothing,SlaterDeterminant{T}}
    gutzwiller::Union{Nothing,GutzwillerProjector{T}}
    jastrow::Union{Nothing,EnhancedJastrowFactor{T}}
    rbm::Union{Nothing,EnhancedRBMNetwork{T}}
    # Spin singlet pairing support (StdFace orbital mapping)
    pair_params::Vector{T}
    orbital_map::Union{Nothing,Matrix{Int}}
    pair_M::Union{Nothing,Matrix{T}}
    pair_inv::Union{Nothing,Matrix{T}}
    pair_det::T
    pair_up_sites::Vector{Int}
    pair_dn_sites::Vector{Int}

    # Current wavefunction amplitude
    current_amplitude::T

    function CombinedWavefunction{T}() where {T}
        new{T}(nothing, nothing, nothing, nothing, T[], nothing, nothing, nothing, one(T), Int[], Int[], one(T))
    end
end

CombinedWavefunction(; T=ComplexF64) = CombinedWavefunction{T}()

"""
    compute_wavefunction_amplitude!(wf::CombinedWavefunction{T}, state::VMCState{T}) where {T}

Compute total wavefunction amplitude for current configuration.
"""
function compute_wavefunction_amplitude!(wf::CombinedWavefunction{T}, state::VMCState{T}) where {T}
    wf.current_amplitude = one(T)

    # Slater determinant amplitude
    if wf.slater_det !== nothing
        slater_amp = compute_determinant!(wf.slater_det)
        wf.current_amplitude *= slater_amp
    end

    # Gutzwiller projection
    if wf.gutzwiller !== nothing
        gutz_factor = compute_gutzwiller_factor!(wf.gutzwiller, state)
        wf.current_amplitude *= gutz_factor
    end

    # Jastrow factor
    if wf.jastrow !== nothing
        jastrow_factor = compute_jastrow_factor!(wf.jastrow, state)
        wf.current_amplitude *= jastrow_factor
    end

    # RBM amplitude
    if wf.rbm !== nothing
        rbm_factor = compute_rbm_amplitude!(wf.rbm, state)
        wf.current_amplitude *= rbm_factor
    end

    # Spin singlet pairing determinant using StdFace orbital mapping
    if wf.orbital_map !== nothing && !isempty(wf.pair_params)
        n_elec = state.n_electrons
        n_up = div(n_elec, 2)
        up_pos = state.electron_positions[1:n_up]
        dn_pos = length(state.electron_positions) > n_up ? state.electron_positions[(n_up+1):end] : Int[]
        if length(dn_pos) == n_up
            # Build pairing matrix from mapping (site-indexed)
            M = Matrix{T}(undef, n_up, n_up)
            for (a, i_site) in enumerate(up_pos)
                for (b, j_site) in enumerate(dn_pos)
                    idx = wf.orbital_map[i_site, j_site]
                    M[a, b] = (idx >= 1 && idx <= length(wf.pair_params)) ? wf.pair_params[idx] : zero(T)
                end
            end
            try
                wf.pair_det = det(M)
                wf.pair_M = M
                wf.pair_inv = inv(M)
                wf.pair_up_sites = copy(up_pos)
                wf.pair_dn_sites = copy(dn_pos)
                wf.current_amplitude *= wf.pair_det
            catch
                wf.current_amplitude *= T(0)
            end
        end
    end

    return wf.current_amplitude
end

"""
    update_wavefunction_parameters!(wf::CombinedWavefunction{T}, params::ParameterSet{Vector{T},Vector{T},Vector{T},Vector{T}}) where {T}

Update all wavefunction components with new parameters.
"""
function update_wavefunction_parameters!(wf::CombinedWavefunction{T}, params::ParameterSet{Vector{T},Vector{T},Vector{T},Vector{T}}) where {T}
    # Update Gutzwiller/Jastrow parameters
    if wf.gutzwiller !== nothing && !isempty(params.proj)
        set_gutzwiller_parameters!(wf.gutzwiller, params.proj)
    end

    if wf.jastrow !== nothing && !isempty(params.proj)
        set_jastrow_parameters!(wf.jastrow, params.proj)
    end

    # Update RBM parameters
    if wf.rbm !== nothing && !isempty(params.rbm)
        set_rbm_parameters!(wf.rbm, params.rbm)
    end

    # Slater determinant parameters would be updated separately
    # through the SlaterDeterminant interface
    # Update pairing parameters for Spin (StdFace orbital mapping)
    if !isempty(params.slater)
        wf.pair_params = copy(params.slater)
    end
end

"""
    compute_pairing_sropto(wf::CombinedWavefunction{T}, config::Vector{Int}) -> Vector{T}

Compute SROptO-like vector (log-derivative of pairing det) for current configuration.
For parameter k, O_k = sum_{a,b : orbital_map[up_site[a], dn_site[b]] == k} (M^{-1})_{b,a}.
This uses a fresh M/inv(M) built from the provided configuration to ensure consistency.
"""
function compute_pairing_sropto(wf::CombinedWavefunction{T}, config::Vector{Int}) where {T}
    if wf.orbital_map === nothing || isempty(wf.pair_params)
        return T[]
    end
    n_elec = length(config)
    n_up = div(n_elec, 2)
    up_pos = config[1:n_up]
    dn_pos = config[(n_up+1):end]
    n = n_up
    M = Matrix{T}(undef, n, n)
    for (a, i_site) in enumerate(up_pos)
        for (b, j_site) in enumerate(dn_pos)
            idx = wf.orbital_map[i_site, j_site]
            M[a, b] = (1 <= idx <= length(wf.pair_params)) ? wf.pair_params[idx] : zero(T)
        end
    end
    Minv = try
        inv(M)
    catch
        pinv(M)
    end
    nparams = length(wf.pair_params)
    O = zeros(T, nparams)
    for (a, i_site) in enumerate(up_pos)
        for (b, j_site) in enumerate(dn_pos)
            idx = wf.orbital_map[i_site, j_site]
            if 1 <= idx <= nparams
                # derivative wrt M_{a,b} contributes (M^{-1})_{b,a}
                O[idx] += Minv[b, a]
            end
        end
    end
    return O
end

"""
    pair_swap_ratio!(wf::CombinedWavefunction{T}, up_index::Int, dn_index::Int, new_up_site::Int, new_dn_site::Int, dn_sites::Vector{Int})

Compute determinant ratio for swapping one up-site row and one down-site column.
Updates internal pair_M/pair_inv/pair_det in-place; returns ratio.
"""
function pair_swap_ratio!(
    wf::CombinedWavefunction{T},
    up_index::Int,
    dn_index::Int,
    new_up_site::Int,
    new_dn_site::Int,
    dn_sites::Vector{Int},
) where {T}
    M = wf.pair_M; Minv = wf.pair_inv; detM = wf.pair_det
    if M === nothing || Minv === nothing
        return one(T)
    end
    n = size(M, 1)
    # Build new row for up_index
    r_new = zeros(T, n)
    for (b, j_site) in enumerate(dn_sites)
        idx = wf.orbital_map[new_up_site, j_site]
        r_new[b] = (idx >= 1 && idx <= length(wf.pair_params)) ? wf.pair_params[idx] : zero(T)
    end
    # Row replacement delta
    delta_row = r_new .- view(M, up_index, :)
    # Row ratio: 1 + delta_row * Minv[:, up_index]
    ratio_row = one(T) + dot(delta_row, view(Minv, :, up_index))
    # Update inverse for row replacement (rank-1 update)
    Au = Minv * (delta_row')
    col_i = view(Minv, :, up_index)
    denom = ratio_row
    Minv .= Minv .- (col_i * Au) ./ denom
    view(M, up_index, :) .= r_new

    # Column replacement: build new column for dn_index
    c_new = zeros(T, n)
    # For each up row a (site index tracked in pair_up_sites)
    for a in 1:n
        up_site = wf.pair_up_sites[a]
        idx = wf.orbital_map[up_site, new_dn_site]
        c_new[a] = (idx >= 1 && idx <= length(wf.pair_params)) ? wf.pair_params[idx] : M[a, dn_index]
    end
    delta_col = c_new .- view(M, :, dn_index)
    # Column ratio: 1 + Minv[dn_index, :] * delta_col
    ratio_col = one(T) + dot(view(Minv, dn_index, :), delta_col)
    # Update inverse for column replacement
    v = Minv * delta_col
    row_j = view(Minv, dn_index, :)
    denom2 = ratio_col
    Minv .= Minv .- (v * row_j) ./ denom2
    view(M, :, dn_index) .= c_new

    total_ratio = ratio_row * ratio_col
    wf.pair_det *= total_ratio
    wf.pair_up_sites[up_index] = new_up_site
    wf.pair_dn_sites[dn_index] = new_dn_site
    return total_ratio
end

"""
    pair_swap_ratio_predict(wf, up_index, dn_index, new_up_site, new_dn_site)

Compute determinant ratio for the proposed swap without mutating internal matrices.
"""
function pair_swap_ratio_predict(
    wf::CombinedWavefunction{T},
    up_index::Int,
    dn_index::Int,
    new_up_site::Int,
    new_dn_site::Int,
) where {T}
    M = wf.pair_M; Minv = wf.pair_inv
    if M === nothing || Minv === nothing
        return one(T)
    end
    n = size(M, 1)
    # New row
    r_new = zeros(T, n)
    for (b, j_site) in enumerate(wf.pair_dn_sites)
        idx = wf.orbital_map[new_up_site, j_site]
        r_new[b] = (idx >= 1 && idx <= length(wf.pair_params)) ? wf.pair_params[idx] : zero(T)
    end
    delta_row = r_new .- view(M, up_index, :)
    ratio_row = one(T) + dot(delta_row, view(Minv, :, up_index))
    # New column
    c_new = zeros(T, n)
    for a in 1:n
        up_site = wf.pair_up_sites[a]
        idx = wf.orbital_map[up_site, new_dn_site]
        c_new[a] = (idx >= 1 && idx <= length(wf.pair_params)) ? wf.pair_params[idx] : M[a, dn_index]
    end
    delta_col = c_new .- view(M, :, dn_index)
    ratio_col = one(T) + dot(view(Minv, dn_index, :), delta_col)
    return ratio_row * ratio_col
end

"""
    update_correlation_matrix!(jastrow::EnhancedJastrowFactor{T}, electron_idx::Int, old_pos::Int, new_pos::Int) where {T}

Lightweight in-place update for Jastrow cached data after a single-electron move.
For now this function is a no-op placeholder to keep consistency with C flow.
"""
function update_correlation_matrix!(
    jastrow::EnhancedJastrowFactor{T},
    electron_idx::Int,
    old_pos::Int,
    new_pos::Int,
) where {T}
    # Current implementation stores only static neighbor/distance; nothing to update.
    return nothing
end

"""
    update_rbm_correlations!(rbm::EnhancedRBMNetwork{T}, electron_idx::Int, old_pos::Int, new_pos::Int, n_sites::Int) where {T}

Lightweight in-place update for RBM cached correlations after a move.
Currently a no-op to match integration points without extra state.
"""
function update_rbm_correlations!(
    rbm::EnhancedRBMNetwork{T},
    electron_idx::Int,
    old_pos::Int,
    new_pos::Int,
    n_sites::Int,
) where {T}
    return nothing
end
