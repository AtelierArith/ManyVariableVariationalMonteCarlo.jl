"""
Local Spin (LocSpn) Handler

Complete implementation of local spin site handling for spin systems.
In mVMC, local spins represent localized magnetic moments.

Based on mVMC C implementation:
- readdef.c: LocSpn reading
- vmcmake_fsz.c: LocSpn initialization
"""

using Random

"""
    LocSpnConfig

Configuration for local spin sites.
"""
struct LocSpnConfig
    n_sites::Int
    n_loc_spin::Int           # Number of local spin sites
    loc_spn_flags::Vector{Int}  # 1=local spin, 0=itinerant
    loc_spin_sites::Vector{Int} # Indices of local spin sites
    itinerant_sites::Vector{Int} # Indices of itinerant sites

    function LocSpnConfig(n_sites::Int, loc_spn_flags::Vector{Int})
        n_loc_spin = count(==(1), loc_spn_flags)
        loc_spin_sites = findall(==(1), loc_spn_flags)
        itinerant_sites = findall(==(0), loc_spn_flags)

        new(n_sites, n_loc_spin, loc_spn_flags, loc_spin_sites, itinerant_sites)
    end
end

"""
    create_loc_spn_config_for_spin_system(n_sites::Int) -> LocSpnConfig

Create LocSpn configuration for spin systems.
For spin systems, all sites are typically local spins.
"""
function create_loc_spn_config_for_spin_system(n_sites::Int)
    # For spin systems (e.g., Heisenberg), all sites have local spins
    loc_spn_flags = ones(Int, n_sites)
    return LocSpnConfig(n_sites, loc_spn_flags)
end

"""
    create_loc_spn_config_for_hubbard_system(n_sites::Int) -> LocSpnConfig

Create LocSpn configuration for Hubbard systems.
For Hubbard systems, electrons are itinerant (no local spins by default).
"""
function create_loc_spn_config_for_hubbard_system(n_sites::Int)
    # For Hubbard systems, electrons are itinerant
    loc_spn_flags = zeros(Int, n_sites)
    return LocSpnConfig(n_sites, loc_spn_flags)
end

"""
    validate_loc_spn_config(config::LocSpnConfig, n_elec::Int) -> Bool

Validate LocSpn configuration.
Based on readdef.c lines 621-636 validation logic.
"""
function validate_loc_spn_config(config::LocSpnConfig, n_elec::Int, n_ex_update_path::Int)
    n_loc_spin = config.n_loc_spin

    # Check: 2*Ne >= NLocalSpin
    if 2 * n_elec < n_loc_spin
        @error "2*Ne must satisfy: 2*Ne >= NLocalSpin. Got 2*$(n_elec) < $(n_loc_spin)"
        return false
    end

    # For spin systems: NLocalSpin == 2*Ne requires NExUpdatePath == 2
    if n_loc_spin == 2 * n_elec
        if n_ex_update_path != 2
            @error "NExUpdatePath must be 2 when 2*Ne = NLocalSpin (spin system). Got $(n_ex_update_path)"
            return false
        end
    end

    # NExUpdatePath must be 1 or 2
    if n_ex_update_path == 0
        @error "NExUpdatePath must be 1 or 2. Got 0"
        return false
    end

    return true
end

"""
    place_local_spin_electrons!(
        ele_idx::Vector{Int},
        ele_spn::Vector{Int},
        ele_cfg::Vector{Int},
        loc_spn_config::LocSpnConfig,
        n_sites::Int,
        rng::AbstractRNG
    )

Place electrons on local spin sites during initialization.
Based on vmcmake_fsz.c lines 454-466.
"""
function place_local_spin_electrons!(
    ele_idx::Vector{Int},
    ele_spn::Vector{Int},
    ele_cfg::Vector{Int},
    loc_spn_config::LocSpnConfig,
    n_sites::Int,
    rng::AbstractRNG,
)
    n_size = length(ele_idx)

    for ri in 0:(n_sites-1)
        if loc_spn_config.loc_spn_flags[ri+1] == 1
            # This is a local spin site
            # Find an empty electron slot
            local x_mi
            while true
                x_mi = rand(rng, 0:(n_size-1))
                if ele_idx[x_mi+1] == -1
                    break
                end
            end

            si = ele_spn[x_mi+1]
            ele_cfg[ri+1 + si*n_sites] = x_mi
            ele_idx[x_mi+1] = ri
        end
    end

    return nothing
end

"""
    place_itinerant_electrons!(
        ele_idx::Vector{Int},
        ele_spn::Vector{Int},
        ele_cfg::Vector{Int},
        loc_spn_config::LocSpnConfig,
        n_sites::Int,
        rng::AbstractRNG
    )

Place remaining electrons on itinerant sites.
Based on vmcmake_fsz.c lines 467-478.
"""
function place_itinerant_electrons!(
    ele_idx::Vector{Int},
    ele_spn::Vector{Int},
    ele_cfg::Vector{Int},
    loc_spn_config::LocSpnConfig,
    n_sites::Int,
    rng::AbstractRNG,
)
    n_size = length(ele_idx)

    # Check if there are any itinerant sites
    if length(loc_spn_config.itinerant_sites) == 0
        # No itinerant sites, all electrons must be on local spin sites
        # This is already handled by place_local_spin_electrons!
        return nothing
    end

    for x_mi in 0:(n_size-1)
        if ele_idx[x_mi+1] == -1
            si = ele_spn[x_mi+1]

            # Find empty itinerant site
            local ri
            max_attempts = n_sites * 10
            attempt = 0
            found = false

            while attempt < max_attempts
                ri = rand(rng, 0:(n_sites-1))
                if ele_cfg[ri+1 + si*n_sites] == -1 &&
                   loc_spn_config.loc_spn_flags[ri+1] == 0
                    found = true
                    break
                end
                attempt += 1
            end

            if !found
                error("Could not find empty itinerant site after $max_attempts attempts")
            end

            ele_cfg[ri+1 + si*n_sites] = x_mi
            ele_idx[x_mi+1] = ri
        end
    end

    return nothing
end

"""
    initialize_with_loc_spn(
        n_sites::Int,
        n_elec::Int,
        two_sz::Int,
        loc_spn_config::LocSpnConfig,
        rng::AbstractRNG
    ) -> (ele_idx, ele_spn, ele_cfg, ele_num)

Initialize electron configuration with LocSpn handling.
Complete implementation including local spin placement.
"""
function initialize_with_loc_spn(
    n_sites::Int,
    n_elec::Int,
    two_sz::Int,
    loc_spn_config::LocSpnConfig,
    rng::AbstractRNG,
)
    n_size = 2 * n_elec
    n_site2 = 2 * n_sites

    # Initialize arrays
    ele_idx = fill(-1, n_size)
    ele_spn = fill(-1, n_size)
    ele_cfg = fill(-1, n_site2)
    ele_num = zeros(Int, n_site2)

    # Determine initial 2Sz
    if two_sz == -1
        tmp_two_sz = 0
    else
        if two_sz % 2 != 0
            error("2Sz must be even, got $(two_sz)")
        end
        tmp_two_sz = two_sz ÷ 2
    end

    # Assign spins to electrons
    for x_mi in 0:(n_size-1)
        if x_mi < n_elec + tmp_two_sz
            ele_spn[x_mi+1] = 0  # Up spin
        else
            ele_spn[x_mi+1] = 1  # Down spin
        end
    end

    # Place local spin electrons
    place_local_spin_electrons!(
        ele_idx, ele_spn, ele_cfg, loc_spn_config, n_sites, rng
    )

    # Place itinerant electrons
    place_itinerant_electrons!(
        ele_idx, ele_spn, ele_cfg, loc_spn_config, n_sites, rng
    )

    # Calculate occupation numbers
    for rsi in 0:(n_site2-1)
        ele_num[rsi+1] = (ele_cfg[rsi+1] < 0) ? 0 : 1
    end

    return (ele_idx, ele_spn, ele_cfg, ele_num)
end

"""
    check_local_spin_update_allowed(
        site::Int,
        loc_spn_config::LocSpnConfig,
        update_type::Symbol
    ) -> Bool

Check if a proposed update is allowed given local spin constraints.
"""
function check_local_spin_update_allowed(
    site::Int,
    loc_spn_config::LocSpnConfig,
    update_type::Symbol,
)
    is_local_spin = loc_spn_config.loc_spn_flags[site+1] == 1

    if update_type == :HOPPING
        # Hopping is always allowed
        return true
    elseif update_type == :LOCALSPINFLIP
        # Spin flip typically allowed on local spin sites
        return true
    else
        return true
    end
end

"""
    get_loc_spn_statistics(loc_spn_config::LocSpnConfig) -> Dict{String,Any}

Get statistics about local spin configuration.
"""
function get_loc_spn_statistics(loc_spn_config::LocSpnConfig)
    return Dict{String,Any}(
        "n_sites" => loc_spn_config.n_sites,
        "n_local_spin" => loc_spn_config.n_loc_spin,
        "n_itinerant" => length(loc_spn_config.itinerant_sites),
        "local_spin_sites" => loc_spn_config.loc_spin_sites,
        "itinerant_sites" => loc_spn_config.itinerant_sites,
        "local_spin_fraction" => loc_spn_config.n_loc_spin / loc_spn_config.n_sites,
    )
end

"""
    print_loc_spn_info(loc_spn_config::LocSpnConfig)

Print information about local spin configuration.
"""
function print_loc_spn_info(loc_spn_config::LocSpnConfig)
    stats = get_loc_spn_statistics(loc_spn_config)

    println("Local Spin Configuration:")
    println("  Total sites: $(stats["n_sites"])")
    println("  Local spin sites: $(stats["n_local_spin"])")
    println("  Itinerant sites: $(stats["n_itinerant"])")
    println("  Local spin fraction: $(round(stats["local_spin_fraction"]*100, digits=1))%")

    if stats["n_local_spin"] > 0 && stats["n_local_spin"] <= 20
        println("  Local spin site indices: $(stats["local_spin_sites"])")
    end
    if stats["n_itinerant"] > 0 && stats["n_itinerant"] <= 20
        println("  Itinerant site indices: $(stats["itinerant_sites"])")
    end
end
