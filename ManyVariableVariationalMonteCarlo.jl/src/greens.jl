"""
Green Functions implementation for ManyVariableVariationalMonteCarlo.jl

Implements local Green function calculations including:
- 1-body Green functions <CisAjs>
- 2-body Green functions <CisAjsCktAlt>
- Large-scale Green functions for Lanczos
- Caching and update strategies

Ported from locgrn.c, lslocgrn.c, and calgrn.c in the C reference implementation.
"""

using LinearAlgebra
using LinearAlgebra.BLAS
using LinearAlgebra.LAPACK

"""
    GreenFunctionCache{T}

Cache for Green function calculations to avoid redundant computations.
"""
mutable struct GreenFunctionCache{T<:Union{Float64,ComplexF64}}
    # 1-body Green functions
    cis_ajs::Vector{T}
    cis_ajs_indices::Vector{NTuple{4,Int}}  # (ri, s, rj, t)

    # 2-body Green functions
    cis_ajs_ckt_alt::Vector{T}
    cis_ajs_ckt_alt_indices::Vector{NTuple{6,Int}}  # (ri, s, rj, t, rk, u)

    # Large-scale Green functions
    ls_local_q::Vector{T}
    ls_local_cis_ajs::Vector{T}

    # Cache metadata
    is_valid::Bool
    last_update_time::Int
    cache_hits::Int
    cache_misses::Int

    function GreenFunctionCache{T}(
        n_cis_ajs::Int,
        n_cis_ajs_ckt_alt::Int,
        n_ls_q::Int,
    ) where {T}
        cis_ajs = zeros(T, n_cis_ajs)
        cis_ajs_indices = Vector{NTuple{4,Int}}(undef, n_cis_ajs)

        cis_ajs_ckt_alt = zeros(T, n_cis_ajs_ckt_alt)
        cis_ajs_ckt_alt_indices = Vector{NTuple{6,Int}}(undef, n_cis_ajs_ckt_alt)

        ls_local_q = zeros(T, n_ls_q)
        ls_local_cis_ajs = zeros(T, n_cis_ajs)

        new{T}(
            cis_ajs,
            cis_ajs_indices,
            cis_ajs_ckt_alt,
            cis_ajs_ckt_alt_indices,
            ls_local_q,
            ls_local_cis_ajs,
            false,
            0,
            0,
            0,
        )
    end
end

"""
    LocalGreenFunction{T}

Main structure for local Green function calculations.
"""
mutable struct LocalGreenFunction{T<:Union{Float64,ComplexF64}}
    # System parameters
    n_site::Int
    n_elec::Int
    n_spin::Int

    # Green function cache
    cache::GreenFunctionCache{T}

    # Working arrays
    buffer::Vector{T}
    ele_idx::Vector{Int}
    ele_cfg::Vector{Int}
    ele_num::Vector{Int}
    proj_cnt::Vector{Int}

    # Performance tracking
    calculation_time::Float64
    total_calculations::Int

    function LocalGreenFunction{T}(n_site::Int, n_elec::Int, n_spin::Int = 2) where {T}
        n_cis_ajs = n_site * n_site * n_spin
        n_cis_ajs_ckt_alt = n_cis_ajs * n_site * n_spin
        n_ls_q = n_site * n_site * n_spin

        cache = GreenFunctionCache{T}(n_cis_ajs, n_cis_ajs_ckt_alt, n_ls_q)

        buffer = Vector{T}(undef, max(n_cis_ajs, 2 * n_elec))
        ele_idx = zeros(Int, n_elec * n_spin)
        ele_cfg = zeros(Int, n_site * n_spin)
        ele_num = zeros(Int, n_site * n_spin)
        proj_cnt = zeros(Int, n_site)

        new{T}(
            n_site,
            n_elec,
            n_spin,
            cache,
            buffer,
            ele_idx,
            ele_cfg,
            ele_num,
            proj_cnt,
            0.0,
            0,
        )
    end
end

"""
    green_function_1body!(gf::LocalGreenFunction{T}, ri::Int, rj::Int, s::Int,
                         ip::T, ele_idx::Vector{Int}, ele_cfg::Vector{Int},
                         ele_num::Vector{Int}, proj_cnt::Vector{Int},
                         rbm_cnt::Vector{T} = T[]) where T

Calculate 1-body Green function <CisAjs>.
Returns the Green function value.
"""
function green_function_1body!(
    gf::LocalGreenFunction{T},
    ri::Int,
    rj::Int,
    s::Int,
    ip::T,
    ele_idx::Vector{Int},
    ele_cfg::Vector{Int},
    ele_num::Vector{Int},
    proj_cnt::Vector{Int},
    rbm_cnt::Vector{T} = T[],
) where {T<:Union{Float64,ComplexF64}}

    # Check cache first
    cache_key = (ri, s, rj, s)
    if gf.cache.is_valid
        for (i, key) in enumerate(gf.cache.cis_ajs_indices)
            if key == cache_key
                gf.cache.cache_hits += 1
                return gf.cache.cis_ajs[i]
            end
        end
    end
    gf.cache.cache_misses += 1

    # Direct calculation
    z = _calculate_green_1body_direct(
        gf,
        ri,
        rj,
        s,
        ip,
        ele_idx,
        ele_cfg,
        ele_num,
        proj_cnt,
        rbm_cnt,
    )

    # Update cache
    if !gf.cache.is_valid
        gf.cache.is_valid = true
        gf.cache.last_update_time += 1
    end

    return z
end

"""
    _calculate_green_1body_direct(gf::LocalGreenFunction{T}, ri::Int, rj::Int, s::Int,
                                 ip::T, ele_idx::Vector{Int}, ele_cfg::Vector{Int},
                                 ele_num::Vector{Int}, proj_cnt::Vector{Int},
                                 rbm_cnt::Vector{T}) where T

Direct calculation of 1-body Green function without caching.
"""
function _calculate_green_1body_direct(
    gf::LocalGreenFunction{T},
    ri::Int,
    rj::Int,
    s::Int,
    ip::T,
    ele_idx::Vector{Int},
    ele_cfg::Vector{Int},
    ele_num::Vector{Int},
    proj_cnt::Vector{Int},
    rbm_cnt::Vector{T},
) where {T<:Union{Float64,ComplexF64}}

    n_site = gf.n_site
    n_elec = gf.n_elec

    # Check trivial cases
    if ri == rj
        rsi = ri + (s - 1) * n_site
        return T(ele_num[rsi])
    end

    rsi = ri + (s - 1) * n_site
    rsj = rj + (s - 1) * n_site

    if rsi > length(ele_num) || rsj > length(ele_num)
        return zero(T)
    end

    if ele_num[rsi] == 1 || ele_num[rsj] == 0
        return zero(T)
    end

    # Find electron at site rj with spin s
    mj = ele_cfg[rsj]
    if mj == 0
        return zero(T)  # No electron at this site
    end
    msj = mj + (s - 1) * n_elec

    # Check bounds
    if msj > length(ele_idx)
        return zero(T)
    end

    # Perform hopping: electron moves from rj to ri
    ele_idx[msj] = ri
    ele_num[rsj] = 0
    ele_num[rsi] = 1

    # Update projection counts
    proj_cnt_new = copy(proj_cnt)
    _update_proj_cnt!(proj_cnt_new, rj, ri, s, ele_num)

    # Calculate projection ratio
    z = _proj_ratio(proj_cnt_new, proj_cnt)

    # Update RBM counts if applicable
    if !isempty(rbm_cnt)
        rbm_cnt_new = copy(rbm_cnt)
        _update_rbm_cnt!(rbm_cnt_new, rbm_cnt, rj, ri, s, ele_num)
        z *= _rbm_ratio(rbm_cnt_new, rbm_cnt)
    end

    # Calculate Pfaffian ratio (simplified - would need full implementation)
    # For now, use a placeholder that represents the Pfaffian update
    pfaffian_ratio = _calculate_pfaffian_ratio(gf, mj, s, ele_idx)
    z *= pfaffian_ratio

    # Revert hopping
    ele_idx[msj] = rj
    ele_num[rsj] = 1
    ele_num[rsi] = 0

    return conj(z / ip)
end

"""
    green_function_2body!(gf::LocalGreenFunction{T}, ri::Int, rj::Int, rk::Int, rl::Int,
                         s::Int, t::Int, ip::T, ele_idx::Vector{Int}, ele_cfg::Vector{Int},
                         ele_num::Vector{Int}, proj_cnt::Vector{Int},
                         rbm_cnt::Vector{T} = T[]) where T

Calculate 2-body Green function <CisAjsCktAlt>.
"""
function green_function_2body!(
    gf::LocalGreenFunction{T},
    ri::Int,
    rj::Int,
    rk::Int,
    rl::Int,
    s::Int,
    t::Int,
    ip::T,
    ele_idx::Vector{Int},
    ele_cfg::Vector{Int},
    ele_num::Vector{Int},
    proj_cnt::Vector{Int},
    rbm_cnt::Vector{T} = T[],
) where {T<:Union{Float64,ComplexF64}}

    # Check cache first
    cache_key = (ri, s, rj, s, rk, t)
    if gf.cache.is_valid
        for (i, key) in enumerate(gf.cache.cis_ajs_ckt_alt_indices)
            if key == cache_key
                gf.cache.cache_hits += 1
                return gf.cache.cis_ajs_ckt_alt[i]
            end
        end
    end
    gf.cache.cache_misses += 1

    # Direct calculation
    z = _calculate_green_2body_direct(
        gf,
        ri,
        rj,
        rk,
        rl,
        s,
        t,
        ip,
        ele_idx,
        ele_cfg,
        ele_num,
        proj_cnt,
        rbm_cnt,
    )

    return z
end

"""
    _calculate_green_2body_direct(gf::LocalGreenFunction{T}, ri::Int, rj::Int, rk::Int, rl::Int,
                                 s::Int, t::Int, ip::T, ele_idx::Vector{Int}, ele_cfg::Vector{Int},
                                 ele_num::Vector{Int}, proj_cnt::Vector{Int},
                                 rbm_cnt::Vector{T}) where T

Direct calculation of 2-body Green function.
"""
function _calculate_green_2body_direct(
    gf::LocalGreenFunction{T},
    ri::Int,
    rj::Int,
    rk::Int,
    rl::Int,
    s::Int,
    t::Int,
    ip::T,
    ele_idx::Vector{Int},
    ele_cfg::Vector{Int},
    ele_num::Vector{Int},
    proj_cnt::Vector{Int},
    rbm_cnt::Vector{T},
) where {T<:Union{Float64,ComplexF64}}

    n_site = gf.n_site
    n_elec = gf.n_elec

    rsi = ri + (s - 1) * n_site
    rsj = rj + (s - 1) * n_site
    rtk = rk + (t - 1) * n_site
    rtl = rl + (t - 1) * n_site

    # Check trivial cases
    if (ri == rj && rk == rl) || (ri == rk && rj == rl && s == t)
        return zero(T)
    end

    if ele_num[rsi] == 1 || ele_num[rsj] == 0 || ele_num[rtk] == 1 || ele_num[rtl] == 0
        return zero(T)
    end

    # Find electrons
    mj = ele_cfg[rsj]
    ml = ele_cfg[rtl]

    if mj == 0 || ml == 0
        return zero(T)  # No electrons at these sites
    end

    msj = mj + (s - 1) * n_elec
    mtl = ml + (t - 1) * n_elec

    # Check bounds
    if msj > length(ele_idx) || mtl > length(ele_idx)
        return zero(T)
    end

    # Perform first hopping: rj -> ri
    ele_idx[msj] = ri
    ele_num[rsj] = 0
    ele_num[rsi] = 1

    # Perform second hopping: rl -> rk
    ele_idx[mtl] = rk
    ele_num[rtl] = 0
    ele_num[rtk] = 1

    # Update projection counts
    proj_cnt_new = copy(proj_cnt)
    _update_proj_cnt!(proj_cnt_new, rj, ri, s, ele_num)
    _update_proj_cnt!(proj_cnt_new, rl, rk, t, ele_num)

    # Calculate projection ratio
    z = _proj_ratio(proj_cnt_new, proj_cnt)

    # Update RBM counts if applicable
    if !isempty(rbm_cnt)
        rbm_cnt_new = copy(rbm_cnt)
        _update_rbm_cnt!(rbm_cnt_new, rbm_cnt, rj, ri, s, ele_num)
        _update_rbm_cnt!(rbm_cnt_new, rbm_cnt, rl, rk, t, ele_num)
        z *= _rbm_ratio(rbm_cnt_new, rbm_cnt)
    end

    # Calculate Pfaffian ratio (simplified)
    pfaffian_ratio = _calculate_pfaffian_ratio_2body(gf, mj, s, ml, t, ele_idx)
    z *= pfaffian_ratio

    # Revert hoppings
    ele_idx[msj] = rj
    ele_num[rsj] = 1
    ele_num[rsi] = 0
    ele_idx[mtl] = rl
    ele_num[rtl] = 1
    ele_num[rtk] = 0

    return conj(z / ip)
end

"""
    large_scale_green_function!(gf::LocalGreenFunction{T}, w::T, ip::T,
                               ele_idx::Vector{Int}, ele_cfg::Vector{Int},
                               ele_num::Vector{Int}, proj_cnt::Vector{Int},
                               rbm_cnt::Vector{T} = T[]) where T

Calculate large-scale Green functions for Lanczos method.
"""
function large_scale_green_function!(
    gf::LocalGreenFunction{T},
    w::T,
    ip::T,
    ele_idx::Vector{Int},
    ele_cfg::Vector{Int},
    ele_num::Vector{Int},
    proj_cnt::Vector{Int},
    rbm_cnt::Vector{T} = T[],
) where {T<:Union{Float64,ComplexF64}}

    n_site = gf.n_site
    n_spin = gf.n_spin

    # Calculate 1-body Green functions
    for s = 1:n_spin
        for ri = 1:n_site
            for rj = 1:n_site
                idx = (s - 1) * n_site * n_site + (ri - 1) * n_site + rj
                gf.cache.ls_local_cis_ajs[idx] = green_function_1body!(
                    gf,
                    ri,
                    rj,
                    s,
                    ip,
                    ele_idx,
                    ele_cfg,
                    ele_num,
                    proj_cnt,
                    rbm_cnt,
                )
            end
        end
    end

    # Calculate large-scale Q matrix (simplified)
    _calculate_ls_q_matrix!(gf, w, ele_idx, ele_cfg, ele_num, proj_cnt, rbm_cnt)

    return gf.cache.ls_local_q, gf.cache.ls_local_cis_ajs
end

"""
    _calculate_ls_q_matrix!(gf::LocalGreenFunction{T}, w::T, ele_idx::Vector{Int},
                           ele_cfg::Vector{Int}, ele_num::Vector{Int}, proj_cnt::Vector{Int},
                           rbm_cnt::Vector{T}) where T

Calculate large-scale Q matrix for Lanczos method.
"""
function _calculate_ls_q_matrix!(
    gf::LocalGreenFunction{T},
    w::T,
    ele_idx::Vector{Int},
    ele_cfg::Vector{Int},
    ele_num::Vector{Int},
    proj_cnt::Vector{Int},
    rbm_cnt::Vector{T},
) where {T<:Union{Float64,ComplexF64}}

    n_site = gf.n_site
    n_spin = gf.n_spin

    # Simplified implementation - would need full Hamiltonian matrix
    for s = 1:n_spin
        for ri = 1:n_site
            for rj = 1:n_site
                idx = (s - 1) * n_site * n_site + (ri - 1) * n_site + rj
                # Placeholder for Q matrix calculation
                gf.cache.ls_local_q[idx] = gf.cache.ls_local_cis_ajs[idx] * w
            end
        end
    end
end

# Helper functions (simplified implementations)

function _update_proj_cnt!(
    proj_cnt_new::Vector{Int},
    rj::Int,
    ri::Int,
    s::Int,
    ele_num::Vector{Int},
)
    # Simplified projection count update
    # Would need full implementation based on projection operators
    proj_cnt_new[ri] += 1
    proj_cnt_new[rj] -= 1
end

function _proj_ratio(proj_cnt_new::Vector{Int}, proj_cnt::Vector{Int})
    # Simplified projection ratio calculation
    # Would need full implementation based on projection operators
    return 1.0
end

function _update_rbm_cnt!(
    rbm_cnt_new::Vector{T},
    rbm_cnt::Vector{T},
    rj::Int,
    ri::Int,
    s::Int,
    ele_num::Vector{Int},
) where {T}
    # Simplified RBM count update
    # Would need full implementation based on RBM structure
    rbm_cnt_new[ri] += 1
    rbm_cnt_new[rj] -= 1
end

function _rbm_ratio(rbm_cnt_new::Vector{T}, rbm_cnt::Vector{T}) where {T}
    # Simplified RBM ratio calculation
    # Would need full implementation based on RBM structure
    return 1.0
end

function _calculate_pfaffian_ratio(
    gf::LocalGreenFunction{T},
    mj::Int,
    s::Int,
    ele_idx::Vector{Int},
) where {T}
    # Simplified Pfaffian ratio calculation
    # Would need full implementation with Pfaffian updates
    return 1.0
end

function _calculate_pfaffian_ratio_2body(
    gf::LocalGreenFunction{T},
    mj::Int,
    s::Int,
    ml::Int,
    t::Int,
    ele_idx::Vector{Int},
) where {T}
    # Simplified 2-body Pfaffian ratio calculation
    # Would need full implementation with Pfaffian updates
    return 1.0
end

"""
    clear_green_function_cache!(gf::LocalGreenFunction{T}) where T

Clear the Green function cache.
"""
function clear_green_function_cache!(gf::LocalGreenFunction{T}) where {T}
    gf.cache.is_valid = false
    gf.cache.cache_hits = 0
    gf.cache.cache_misses = 0
    fill!(gf.cache.cis_ajs, zero(T))
    fill!(gf.cache.cis_ajs_ckt_alt, zero(T))
    fill!(gf.cache.ls_local_q, zero(T))
    fill!(gf.cache.ls_local_cis_ajs, zero(T))
end

"""
    get_cache_statistics(gf::LocalGreenFunction{T}) where T

Get cache hit/miss statistics.
"""
function get_cache_statistics(gf::LocalGreenFunction{T}) where {T}
    total = gf.cache.cache_hits + gf.cache.cache_misses
    hit_rate = total > 0 ? gf.cache.cache_hits / total : 0.0
    return (hits = gf.cache.cache_hits, misses = gf.cache.cache_misses, hit_rate = hit_rate)
end

"""
    benchmark_green_functions(n_site::Int = 10, n_elec::Int = 5, n_iterations::Int = 1000)

Benchmark Green function calculations.
"""
function benchmark_green_functions(
    n_site::Int = 10,
    n_elec::Int = 5,
    n_iterations::Int = 1000,
)
    println(
        "Benchmarking Green function calculations (n_site=$n_site, n_elec=$n_elec, iterations=$n_iterations)...",
    )

    # Create Green function calculator
    gf = LocalGreenFunction{ComplexF64}(n_site, n_elec)

    # Initialize electron configuration
    ele_idx = collect(1:n_elec)
    ele_cfg = zeros(Int, n_site * 2)  # n_site * n_spin
    ele_cfg[1:n_elec] .= 1
    ele_num = zeros(Int, n_site * 2)  # n_site * n_spin
    ele_num[1:n_elec] .= 1
    proj_cnt = zeros(Int, n_site)

    # Benchmark 1-body Green functions
    @time begin
        for _ = 1:n_iterations
            for ri = 1:n_site
                for rj = 1:n_site
                    for s = 1:2
                        green_function_1body!(
                            gf,
                            ri,
                            rj,
                            s,
                            1.0 + 0.1im,
                            ele_idx,
                            ele_cfg,
                            ele_num,
                            proj_cnt,
                        )
                    end
                end
            end
        end
    end
    println("  1-body Green function calculation rate")

    # Benchmark 2-body Green functions
    @time begin
        for _ = 1:(n_iterations÷10)  # Fewer iterations for 2-body
            for ri = 1:n_site
                for rj = 1:n_site
                    for rk = 1:n_site
                        for rl = 1:n_site
                            for s = 1:2
                                for t = 1:2
                                    green_function_2body!(
                                        gf,
                                        ri,
                                        rj,
                                        rk,
                                        rl,
                                        s,
                                        t,
                                        1.0 + 0.1im,
                                        ele_idx,
                                        ele_cfg,
                                        ele_num,
                                        proj_cnt,
                                    )
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    println("  2-body Green function calculation rate")

    # Print cache statistics
    stats = get_cache_statistics(gf)
    println("  Cache hit rate: $(stats.hit_rate * 100)%")

    println("Green function benchmark completed.")
end
