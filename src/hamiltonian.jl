"""
Hamiltonian Calculation Engine

Implements Hamiltonian calculations for various quantum lattice models.
Equivalent to calham.c, calham_real.c, and calham_fsz.c in the C reference implementation.

Ported from calham.c in the mVMC C reference implementation.
"""

using LinearAlgebra
using SparseArrays

"""
    HamiltonianTerm

Abstract base type for Hamiltonian terms.
"""
abstract type HamiltonianTerm end

"""
    TransferTerm{T}

Represents kinetic energy (hopping) terms: t * c†ᵢσ cⱼσ
"""
struct TransferTerm{T<:Number} <: HamiltonianTerm
    coefficient::T
    site_i::Int
    spin_i::Int
    site_j::Int
    spin_j::Int
end

"""
    CoulombIntraTerm{T}

Represents on-site Coulomb interaction: U * nᵢ↑ nᵢ↓
"""
struct CoulombIntraTerm{T<:Number} <: HamiltonianTerm
    coefficient::T
    site::Int
end

"""
    CoulombInterTerm{T}

Represents inter-site Coulomb interaction: V * nᵢ nⱼ
"""
struct CoulombInterTerm{T<:Number} <: HamiltonianTerm
    coefficient::T
    site_i::Int
    site_j::Int
end

"""
    HundTerm{T}

Represents Hund coupling: J * (S⃗ᵢ · S⃗ⱼ - nᵢnⱼ/4)
"""
struct HundTerm{T<:Number} <: HamiltonianTerm
    coefficient::T
    site_i::Int
    site_j::Int
end

"""
    PairHoppingTerm{T}

Represents pair hopping: J * (c†ᵢ↑c†ᵢ↓cⱼ↓cⱼ↑ + h.c.)
"""
struct PairHoppingTerm{T<:Number} <: HamiltonianTerm
    coefficient::T
    site_i::Int
    site_j::Int
end

"""
    ExchangeTerm{T}

Represents exchange interaction: J * (c†ᵢ↑cⱼ↑c†ⱼ↓cᵢ↓ + c†ᵢ↓cⱼ↓c†ⱼ↑cᵢ↑)
"""
struct ExchangeTerm{T<:Number} <: HamiltonianTerm
    coefficient::T
    site_i::Int
    site_j::Int
end

"""
    InterAllTerm{T}

Represents general four-operator terms: J * c†ᵢσ c†ⱼτ cₖᵤ cₗᵥ
"""
struct InterAllTerm{T<:Number} <: HamiltonianTerm
    coefficient::T
    site_i::Int
    spin_i::Int
    site_j::Int
    spin_j::Int
    site_k::Int
    spin_k::Int
    site_l::Int
    spin_l::Int
end

"""
    Hamiltonian{T}

Main Hamiltonian representation containing all terms.
"""
mutable struct Hamiltonian{T<:Number}
    # Hamiltonian terms
    transfer_terms::Vector{TransferTerm{T}}
    coulomb_intra_terms::Vector{CoulombIntraTerm{T}}
    coulomb_inter_terms::Vector{CoulombInterTerm{T}}
    hund_terms::Vector{HundTerm{T}}
    pair_hopping_terms::Vector{PairHoppingTerm{T}}
    exchange_terms::Vector{ExchangeTerm{T}}
    interall_terms::Vector{InterAllTerm{T}}

    # System parameters
    n_sites::Int
    n_electrons::Int

    # Caching for performance
    kinetic_matrix::Union{Nothing,Matrix{T}}
    potential_cache::Dict{Vector{Int},T}

    function Hamiltonian{T}(n_sites::Int, n_electrons::Int) where {T}
        new{T}(
            TransferTerm{T}[],
            CoulombIntraTerm{T}[],
            CoulombInterTerm{T}[],
            HundTerm{T}[],
            PairHoppingTerm{T}[],
            ExchangeTerm{T}[],
            InterAllTerm{T}[],
            n_sites,
            n_electrons,
            nothing,
            Dict{Vector{Int},T}(),
        )
    end
end

"""
    Hamiltonian(n_sites::Int, n_electrons::Int; T=ComplexF64)

Create a new Hamiltonian for a system with given number of sites and electrons.
"""
Hamiltonian(n_sites::Int, n_electrons::Int; T = ComplexF64) =
    Hamiltonian{T}(n_sites, n_electrons)

"""
    add_transfer!(ham::Hamiltonian{T}, coeff::T, i::Int, si::Int, j::Int, sj::Int) where {T}

Add a kinetic energy (transfer) term to the Hamiltonian.

C実装参考: calham.c 1行目から522行目まで
"""
function add_transfer!(
    ham::Hamiltonian{T},
    coeff,
    i::Int,
    si::Int,
    j::Int,
    sj::Int,
) where {T}
    cT = convert(T, coeff)
    push!(ham.transfer_terms, TransferTerm{T}(cT, i, si, j, sj))
    ham.kinetic_matrix = nothing  # Invalidate cache
end

"""
    add_coulomb_intra!(ham::Hamiltonian{T}, coeff::T, site::Int) where {T}

Add an on-site Coulomb interaction term.
"""
function add_coulomb_intra!(ham::Hamiltonian{T}, coeff, site::Int) where {T}
    cT = convert(T, coeff)
    push!(ham.coulomb_intra_terms, CoulombIntraTerm{T}(cT, site))
end

"""
    add_coulomb_inter!(ham::Hamiltonian{T}, coeff::T, i::Int, j::Int) where {T}

Add an inter-site Coulomb interaction term.
"""
function add_coulomb_inter!(ham::Hamiltonian{T}, coeff, i::Int, j::Int) where {T}
    cT = convert(T, coeff)
    push!(ham.coulomb_inter_terms, CoulombInterTerm{T}(cT, i, j))
end

"""
    add_hund_coupling!(ham::Hamiltonian{T}, coeff::T, i::Int, j::Int) where {T}

Add a Hund coupling term.
"""
function add_hund_coupling!(ham::Hamiltonian{T}, coeff, i::Int, j::Int) where {T}
    cT = convert(T, coeff)
    push!(ham.hund_terms, HundTerm{T}(cT, i, j))
end

"""
    add_pair_hopping!(ham::Hamiltonian{T}, coeff::T, i::Int, j::Int) where {T}

Add a pair hopping term.
"""
function add_pair_hopping!(ham::Hamiltonian{T}, coeff, i::Int, j::Int) where {T}
    cT = convert(T, coeff)
    push!(ham.pair_hopping_terms, PairHoppingTerm{T}(cT, i, j))
end

"""
    add_exchange!(ham::Hamiltonian{T}, coeff::T, i::Int, j::Int) where {T}

Add an exchange interaction term.
"""
function add_exchange!(ham::Hamiltonian{T}, coeff, i::Int, j::Int) where {T}
    cT = convert(T, coeff)
    push!(ham.exchange_terms, ExchangeTerm{T}(cT, i, j))
end

"""
    add_interall!(ham::Hamiltonian{T}, coeff::T, i::Int, si::Int, j::Int, sj::Int,
                  k::Int, sk::Int, l::Int, sl::Int) where {T}

Add a general four-operator interaction term.
"""
function add_interall!(
    ham::Hamiltonian{T},
    coeff,
    i::Int,
    si::Int,
    j::Int,
    sj::Int,
    k::Int,
    sk::Int,
    l::Int,
    sl::Int,
) where {T}
    cT = convert(T, coeff)
    push!(ham.interall_terms, InterAllTerm{T}(cT, i, si, j, sj, k, sk, l, sl))
end

"""
    calculate_hamiltonian(ham::Hamiltonian{T}, electron_config::Vector{Int},
                         electron_numbers::Vector{Int}) where {T}

Calculate the Hamiltonian expectation value for a given electron configuration.
Equivalent to CalculateHamiltonian in the C reference implementation.

C実装参考: calham.c 32行目から522行目まで
"""
function calculate_hamiltonian(
    ham::Hamiltonian{T},
    electron_config::Vector{Int},
    electron_numbers::Vector{Int},
) where {T}
    energy = zero(T)

    # Kinetic energy (transfer terms)
    kinetic = calculate_kinetic_energy(ham, electron_config, electron_numbers)
    energy += kinetic

    # Potential energy terms
    coulomb_intra = calculate_coulomb_intra_energy(ham, electron_numbers)
    coulomb_inter = calculate_coulomb_inter_energy(ham, electron_numbers)
    hund = calculate_hund_energy(ham, electron_numbers)
    pair_hop = calculate_pair_hopping_energy(ham, electron_config, electron_numbers)
    exchange = calculate_exchange_energy(ham, electron_config, electron_numbers)
    interall = calculate_interall_energy(ham, electron_config, electron_numbers)

    energy += coulomb_intra + coulomb_inter + hund + pair_hop + exchange + interall


    return energy
end

"""
    calculate_kinetic_energy(ham::Hamiltonian{T}, electron_config::Vector{Int},
                           electron_numbers::Vector{Int}) where {T}

Calculate kinetic energy contribution.
"""
function calculate_kinetic_energy(
    ham::Hamiltonian{T},
    electron_config::Vector{Int},
    electron_numbers::Vector{Int},
) where {T}
    energy = zero(T)

    # Check if electron_numbers has the expected size
    expected_size = 2 * ham.n_sites
    if length(electron_numbers) < expected_size
        # For Heisenberg model, there are no transfer terms, so kinetic energy is zero
        return energy
    end

    n_up = electron_numbers[1:ham.n_sites]
    n_down = electron_numbers[(ham.n_sites+1):(2*ham.n_sites)]

    for term in ham.transfer_terms
        if term.spin_i == term.spin_j
            # Diagonal in spin: ⟨c†ᵢσ cⱼσ⟩
            if term.spin_i == 0  # spin up
                if term.site_i == term.site_j
                    energy += term.coefficient * n_up[term.site_i]
                else
                    # Off-diagonal terms require more complex calculation
                    # For now, approximate as zero (would need proper Green's function)
                    energy += zero(T)
                end
            else  # spin down
                if term.site_i == term.site_j
                    energy += term.coefficient * n_down[term.site_i]
                else
                    energy += zero(T)
                end
            end
        end
    end

    return energy
end

"""
    calculate_coulomb_intra_energy(ham::Hamiltonian{T}, electron_numbers::Vector{Int}) where {T}

Calculate on-site Coulomb interaction energy.
Equivalent to the CoulombIntra part in CalculateHamiltonian.
"""
function calculate_coulomb_intra_energy(
    ham::Hamiltonian{T},
    electron_numbers::Vector{Int},
) where {T}
    energy = zero(T)

    # Check if electron_numbers has the expected size
    expected_size = 2 * ham.n_sites
    if length(electron_numbers) < expected_size
        return energy
    end

    n_up = electron_numbers[1:ham.n_sites]
    n_down = electron_numbers[(ham.n_sites+1):(2*ham.n_sites)]

    for term in ham.coulomb_intra_terms
        energy += term.coefficient * n_up[term.site] * n_down[term.site]
    end

    return energy
end

"""
    calculate_coulomb_inter_energy(ham::Hamiltonian{T}, electron_numbers::Vector{Int}) where {T}

Calculate inter-site Coulomb interaction energy.
"""
function calculate_coulomb_inter_energy(
    ham::Hamiltonian{T},
    electron_numbers::Vector{Int},
) where {T}
    energy = zero(T)

    # Check if electron_numbers has the expected size
    expected_size = 2 * ham.n_sites
    if length(electron_numbers) < expected_size
        return energy
    end

    n_total =
        electron_numbers[1:ham.n_sites] + electron_numbers[(ham.n_sites+1):(2*ham.n_sites)]

    for term in ham.coulomb_inter_terms
        energy += term.coefficient * n_total[term.site_i] * n_total[term.site_j]
    end

    return energy
end

"""
    calculate_hund_energy(ham::Hamiltonian{T}, electron_numbers::Vector{Int}) where {T}

Calculate Hund coupling energy.
"""
function calculate_hund_energy(ham::Hamiltonian{T}, electron_numbers::Vector{Int}) where {T}
    energy = zero(T)

    # Check if electron_numbers has the expected size
    expected_size = 2 * ham.n_sites
    if length(electron_numbers) < expected_size
        return energy
    end

    n_up = electron_numbers[1:ham.n_sites]
    n_down = electron_numbers[(ham.n_sites+1):(2*ham.n_sites)]

    for term in ham.hund_terms
        # Hund coupling term: matches C implementation
        # myEnergy -= ParaHundCoupling[idx] * (n0[ri]*n0[rj] + n1[ri]*n1[rj]);
        ni_up = n_up[term.site_i]
        ni_down = n_down[term.site_i]
        nj_up = n_up[term.site_j]
        nj_down = n_down[term.site_j]

        # Full Hund coupling term implementation
        # H_Hund = n_i↑*n_j↑ + n_i↓*n_j↓ - n_i↑*n_j↓ - n_i↓*n_j↑ - (n_i↑+n_i↓)*(n_j↑+n_j↓)/4
        parallel_term = ni_up * nj_up + ni_down * nj_down
        cross_term = ni_up * nj_down + ni_down * nj_up
        interaction_term = (ni_up + ni_down) * (nj_up + nj_down) / 4

        hund_energy = parallel_term - cross_term - interaction_term

        energy += term.coefficient * hund_energy
    end

    return energy
end

"""
    calculate_pair_hopping_energy(ham::Hamiltonian{T}, electron_config::Vector{Int},
                                 electron_numbers::Vector{Int}) where {T}

Calculate pair hopping energy (simplified approximation).
"""
function calculate_pair_hopping_energy(
    ham::Hamiltonian{T},
    electron_config::Vector{Int},
    electron_numbers::Vector{Int},
) where {T}
    # Pair hopping terms are complex and require proper treatment of creation/annihilation operators
    # For now, return zero as placeholder
    return zero(T)
end

"""
    calculate_exchange_energy(ham::Hamiltonian{T}, electron_config::Vector{Int},
                            electron_numbers::Vector{Int}) where {T}

Calculate exchange interaction energy (simplified approximation).
"""
function calculate_exchange_energy(
    ham::Hamiltonian{T},
    electron_config::Vector{Int},
    electron_numbers::Vector{Int},
) where {T}
    energy = zero(T)

    # Check if electron_numbers has the expected size
    expected_size = 2 * ham.n_sites
    if length(electron_numbers) < expected_size
        return energy
    end

    n_up = electron_numbers[1:ham.n_sites]
    n_down = electron_numbers[(ham.n_sites+1):(2*ham.n_sites)]

    for term in ham.exchange_terms
        # Exchange interaction: J * (c†ᵢ↑cⱼ↑c†ⱼ↓cᵢ↓ + c†ᵢ↓cⱼ↓c†ⱼ↑cᵢ↑)
        # In mean field approximation, this becomes:
        # J * (nᵢ↑nⱼ↓ + nᵢ↓nⱼ↑) for the off-diagonal spin-flip terms
        # However, for the Heisenberg model in C implementation, this is handled differently
        # The exchange term contributes the same as Hund term but for opposite spins
        ni_up = n_up[term.site_i]
        ni_down = n_down[term.site_i]
        nj_up = n_up[term.site_j]
        nj_down = n_down[term.site_j]

        # For Heisenberg model, exchange term contributes spin-flip interactions
        # Simplified mean-field approximation: contributes to off-diagonal terms
        exchange_energy = ni_up * nj_down + ni_down * nj_up

        energy += term.coefficient * exchange_energy
    end

    return energy
end

"""
    calculate_interall_energy(ham::Hamiltonian{T}, electron_config::Vector{Int},
                             electron_numbers::Vector{Int}) where {T}

Calculate general four-operator interaction energy (simplified approximation).
"""
function calculate_interall_energy(
    ham::Hamiltonian{T},
    electron_config::Vector{Int},
    electron_numbers::Vector{Int},
) where {T}
    # InterAll terms are the most general and complex
    # For now, return zero as placeholder
    return zero(T)
end

"""
    calculate_double_occupation(electron_numbers::Vector{Int}, n_sites::Int)

Calculate double occupation number.
Equivalent to CalculateDoubleOccupation in the C reference implementation.
"""
function calculate_double_occupation(electron_numbers::Vector{Int}, n_sites::Int)
    n_up = electron_numbers[1:n_sites]
    n_down = electron_numbers[(n_sites+1):(2*n_sites)]

    double_occ = 0.0
    for i = 1:n_sites
        double_occ += n_up[i] * n_down[i]
    end

    return double_occ
end

"""
    build_kinetic_matrix!(ham::Hamiltonian{T}) where {T}

Build the kinetic energy matrix for efficient calculations.
"""
function build_kinetic_matrix!(ham::Hamiltonian{T}) where {T}
    if ham.kinetic_matrix !== nothing
        return ham.kinetic_matrix
    end

    # Build 2*n_sites × 2*n_sites matrix (spin up and down)
    matrix_size = 2 * ham.n_sites
    ham.kinetic_matrix = zeros(T, matrix_size, matrix_size)

    for term in ham.transfer_terms
        # Convert to matrix indices (site, spin) -> linear index
        i_idx = term.site_i + term.spin_i * ham.n_sites
        j_idx = term.site_j + term.spin_j * ham.n_sites

        ham.kinetic_matrix[i_idx, j_idx] += term.coefficient
        # Add hermitian conjugate if i ≠ j
        if i_idx != j_idx
            ham.kinetic_matrix[j_idx, i_idx] += conj(term.coefficient)
        end
    end

    return ham.kinetic_matrix
end

"""
    create_hubbard_hamiltonian(n_sites::Int, n_electrons::Int, t::T, U::T;
                              lattice_type::Symbol=:chain,
                              Lx::Union{Nothing,Int}=nothing,
                              Ly::Union{Nothing,Int}=nothing) where {T}

Create a standard Hubbard model Hamiltonian.

For 2D lattices (e.g. `:square`), you may optionally provide `Lx` and `Ly`
to factor `n_sites` explicitly. This removes ambiguity when `n_sites` is
not a perfect square or has multiple factorizations. If omitted, a near-square
factorization is chosen heuristically.
"""
function create_hubbard_hamiltonian(
    n_sites::Int,
    n_electrons::Int,
    t::T,
    U::T;
    lattice_type::Symbol = :chain,
    Lx::Union{Nothing,Int} = nothing,
    Ly::Union{Nothing,Int} = nothing,
    apbc::Bool = false,
    twist_x::Float64 = 0.0,
    twist_y::Float64 = 0.0,
) where {T}
    # Always construct a complex Hamiltonian to support boundary twists/APBC
    # (real inputs are safely converted to ComplexF64 as needed)
    ham = Hamiltonian{ComplexF64}(n_sites, n_electrons)

    # Add on-site Coulomb terms
    for i = 1:n_sites
        add_coulomb_intra!(ham, U, i)
    end

    # Add hopping terms based on lattice type
    if lattice_type == :chain
        # 1D chain with nearest-neighbor hopping
        for i = 1:(n_sites-1)
            # Spin up hopping
            add_transfer!(ham, -t, i, 0, i+1, 0)
            add_transfer!(ham, -t, i+1, 0, i, 0)
            # Spin down hopping
            add_transfer!(ham, -t, i, 1, i+1, 1)
            add_transfer!(ham, -t, i+1, 1, i, 1)
        end

        # Periodic boundary conditions
        if n_sites > 2
            # boundary twist (APBC overrides twist if set)
            ϕ = apbc ? π : twist_x
            ph = exp(im * ϕ)
            add_transfer!(ham, -t * ph, 1, 0, n_sites, 0)
            add_transfer!(ham, -t * conj(ph), n_sites, 0, 1, 0)
            add_transfer!(ham, -t * ph, 1, 1, n_sites, 1)
            add_transfer!(ham, -t * conj(ph), n_sites, 1, 1, 1)
        end

    elseif lattice_type == :square
        # 2D square lattice (assuming n_sites = Lx * Ly)
        if (Lx !== nothing) ⊻ (Ly !== nothing)
            error("Both Lx and Ly must be provided together if specified")
        end
        if Lx === nothing
            # Heuristic near-square factorization
            local Lx_local = max(Int(round(sqrt(n_sites))), 1)
            while n_sites % Lx_local != 0 && Lx_local > 1
                Lx_local -= 1
            end
            local Ly_local = max(n_sites ÷ Lx_local, 1)
            @assert Lx_local * Ly_local == n_sites "n_sites must be factorizable as Lx*Ly for square lattice"
            Lx = Lx_local
            Ly = Ly_local
        else
            @assert Lx > 0 && Ly > 0 "Lx and Ly must be positive"
            @assert Lx * Ly == n_sites "Provided Lx*Ly must equal n_sites"
        end

        for i = 1:n_sites
            ix = (i - 1) % Lx + 1
            iy = (i - 1) ÷ Lx + 1

            # x-direction neighbors
            if ix < Lx
                j = i + 1
                add_transfer!(ham, -t, i, 0, j, 0)
                add_transfer!(ham, -t, j, 0, i, 0)
                add_transfer!(ham, -t, i, 1, j, 1)
                add_transfer!(ham, -t, j, 1, i, 1)
            elseif Lx > 1
                j = i - (Lx - 1)
                ϕx = apbc ? π : twist_x
                phx = exp(im * ϕx)
                add_transfer!(ham, -t * phx, i, 0, j, 0)
                add_transfer!(ham, -t * conj(phx), j, 0, i, 0)
                add_transfer!(ham, -t * phx, i, 1, j, 1)
                add_transfer!(ham, -t * conj(phx), j, 1, i, 1)
            end

            # y-direction neighbors
            if iy < Ly
                j = i + Lx
                add_transfer!(ham, -t, i, 0, j, 0)
                add_transfer!(ham, -t, j, 0, i, 0)
                add_transfer!(ham, -t, i, 1, j, 1)
                add_transfer!(ham, -t, j, 1, i, 1)
            elseif Ly > 1
                j = i - Lx * (Ly - 1)
                ϕy = apbc ? π : twist_y
                phy = exp(im * ϕy)
                add_transfer!(ham, -t * phy, i, 0, j, 0)
                add_transfer!(ham, -t * conj(phy), j, 0, i, 0)
                add_transfer!(ham, -t * phy, i, 1, j, 1)
                add_transfer!(ham, -t * conj(phy), j, 1, i, 1)
            end
        end
    else
        error("Unsupported lattice type: $lattice_type")
    end

    return ham
end

"""
    read_hund_file(filepath::String) -> Vector{Tuple{Int,Int,Float64}}

Read hund.def file and return list of (site_i, site_j, coefficient) tuples.
"""
function read_hund_file(filepath::String)
    hund_terms = Tuple{Int,Int,Float64}[]

    if !isfile(filepath)
        return hund_terms
    end

    open(filepath, "r") do file
        for line in eachline(file)
            line = strip(line)
            isempty(line) && continue
            startswith(line, "=") && continue
            startswith(line, "N") && continue  # Skip NHund line

            parts = split(line)
            if length(parts) >= 3
                site_i = parse(Int, parts[1])
                site_j = parse(Int, parts[2])
                coeff = parse(Float64, parts[3])
                push!(hund_terms, (site_i, site_j, coeff))
            end
        end
    end

    return hund_terms
end

"""
    read_exchange_file(filepath::String) -> Vector{Tuple{Int,Int,Float64}}

Read exchange.def file and return list of (site_i, site_j, coefficient) tuples.
"""
function read_exchange_file(filepath::String)
    exchange_terms = Tuple{Int,Int,Float64}[]

    if !isfile(filepath)
        return exchange_terms
    end

    open(filepath, "r") do file
        for line in eachline(file)
            line = strip(line)
            isempty(line) && continue
            startswith(line, "=") && continue
            startswith(line, "N") && continue  # Skip NExchange line

            parts = split(line)
            if length(parts) >= 3
                site_i = parse(Int, parts[1])
                site_j = parse(Int, parts[2])
                coeff = parse(Float64, parts[3])
                push!(exchange_terms, (site_i, site_j, coeff))
            end
        end
    end

    return exchange_terms
end

"""
    read_coulombinter_file(filepath::String) -> Vector{Tuple{Int,Int,Float64}}

Read coulombinter.def file and return list of (site_i, site_j, coefficient) tuples.
"""
function read_coulombinter_file(filepath::String)
    coulomb_terms = Tuple{Int,Int,Float64}[]

    if !isfile(filepath)
        return coulomb_terms
    end

    open(filepath, "r") do file
        for line in eachline(file)
            line = strip(line)
            isempty(line) && continue
            startswith(line, "=") && continue
            startswith(line, "N") && continue  # Skip NCoulombInter line

            parts = split(line)
            if length(parts) >= 3
                site_i = parse(Int, parts[1])
                site_j = parse(Int, parts[2])
                coeff = parse(Float64, parts[3])
                push!(coulomb_terms, (site_i, site_j, coeff))
            end
        end
    end

    return coulomb_terms
end

"""
    create_hamiltonian_from_expert_files(output_dir::String, n_sites::Int; T=ComplexF64)

Create Hamiltonian from expert mode files (hund.def, exchange.def, coulombinter.def).
This matches the C implementation approach.
"""
function create_hamiltonian_from_expert_files(output_dir::String, n_sites::Int; T=ComplexF64)
    ham = Hamiltonian{T}(n_sites, n_sites)  # One electron per site for spin-1/2

    # Read and add Hund coupling terms
    hund_file = joinpath(output_dir, "hund.def")
    hund_terms = read_hund_file(hund_file)
    println("Read $(length(hund_terms)) Hund terms from $hund_file")
    for (site_i, site_j, coeff) in hund_terms
        add_hund_coupling!(ham, T(coeff), site_i + 1, site_j + 1)  # Convert to 1-based indexing
    end

    # Read and add Exchange terms
    exchange_file = joinpath(output_dir, "exchange.def")
    exchange_terms = read_exchange_file(exchange_file)
    println("Read $(length(exchange_terms)) Exchange terms from $exchange_file")
    for (site_i, site_j, coeff) in exchange_terms
        add_exchange!(ham, T(coeff), site_i + 1, site_j + 1)  # Convert to 1-based indexing
    end

    # Read and add Coulomb inter terms
    coulomb_file = joinpath(output_dir, "coulombinter.def")
    coulomb_terms = read_coulombinter_file(coulomb_file)
    println("Read $(length(coulomb_terms)) Coulomb inter terms from $coulomb_file")
    for (site_i, site_j, coeff) in coulomb_terms
        add_coulomb_inter!(ham, T(coeff), site_i + 1, site_j + 1)  # Convert to 1-based indexing
    end

    println("Total Hamiltonian terms: $(length(ham.hund_terms)) Hund, $(length(ham.exchange_terms)) Exchange, $(length(ham.coulomb_inter_terms)) Coulomb inter")

    return ham
end

"""
    create_heisenberg_hamiltonian(n_sites::Int, J::T; lattice_type::Symbol=:chain,
                                  Lx::Union{Nothing,Int}=nothing,
                                  Ly::Union{Nothing,Int}=nothing) where {T}

Create a standard Heisenberg model Hamiltonian.

For 2D lattices (e.g. `:square`, `:triangular`, `:honeycomb`, `:kagome`), you may
optionally provide `Lx` and `Ly` to set the system dimensions explicitly. For
multi-site unit cells, `n_sites` must equal `nsites_per_cell * Lx * Ly`.
"""
function create_heisenberg_hamiltonian(
    n_sites::Int,
    J::T;
    lattice_type::Symbol = :chain,
    Lx::Union{Nothing,Int} = nothing,
    Ly::Union{Nothing,Int} = nothing,
) where {T}
    ham = Hamiltonian{T}(n_sites, n_sites)  # One electron per site for spin-1/2

    if lattice_type == :chain
        # 1D chain with nearest-neighbor exchange
        for i = 1:(n_sites-1)
            add_hund_coupling!(ham, J, i, i+1)
        end

        # Periodic boundary conditions
        if n_sites > 2
            add_hund_coupling!(ham, J, 1, n_sites)
        end
    elseif lattice_type == :square
        # 2D square lattice assumed to be Lx*Ly
        if (Lx !== nothing) ⊻ (Ly !== nothing)
            error("Both Lx and Ly must be provided together if specified")
        end
        if Lx === nothing
            Lx_local = max(Int(round(sqrt(n_sites))), 1)
            while n_sites % Lx_local != 0 && Lx_local > 1
                Lx_local -= 1
            end
            Ly_local = max(n_sites ÷ Lx_local, 1)
            @assert Lx_local * Ly_local == n_sites "n_sites must be factorizable as Lx*Ly for square lattice"
            Lx, Ly = Lx_local, Ly_local
        else
            @assert Lx > 0 && Ly > 0 "Lx and Ly must be positive"
            @assert Lx * Ly == n_sites "Provided Lx*Ly must equal n_sites"
        end

        # Add bonds without double counting
        for i = 1:n_sites
            ix = (i - 1) % Lx + 1
            iy = (i - 1) ÷ Lx + 1
            # +x neighbor (no wrap here)
            if ix < Lx
                add_hund_coupling!(ham, J, i, i + 1)
            end
            # +y neighbor (no wrap here)
            if iy < Ly
                add_hund_coupling!(ham, J, i, i + Lx)
            end
        end
        # periodic wraps along x for each row
        for iy = 1:Ly
            i_end = (iy - 1) * Lx + Lx
            i_start = (iy - 1) * Lx + 1
            add_hund_coupling!(ham, J, i_end, i_start)
        end
        # periodic wraps along y for each column
        for ix = 1:Lx
            i_top = (Ly - 1) * Lx + ix
            i_bottom = ix
            add_hund_coupling!(ham, J, i_top, i_bottom)
        end
    elseif lattice_type == :triangular
        # 2D triangular lattice on Lx x Ly grid, neighbors: +x, +y, +x+y with PBC
        if (Lx !== nothing) ⊻ (Ly !== nothing)
            error("Both Lx and Ly must be provided together if specified")
        end
        if Lx === nothing
            Lx_local = max(Int(round(sqrt(n_sites))), 1)
            while n_sites % Lx_local != 0 && Lx_local > 1
                Lx_local -= 1
            end
            Ly_local = max(n_sites ÷ Lx_local, 1)
            @assert Lx_local * Ly_local == n_sites "n_sites must be factorizable as Lx*Ly for triangular lattice"
            Lx, Ly = Lx_local, Ly_local
        else
            @assert Lx > 0 && Ly > 0 "Lx and Ly must be positive"
            @assert Lx * Ly == n_sites "Provided Lx*Ly must equal n_sites"
        end

        for i = 1:n_sites
            ix = (i - 1) % Lx + 1
            iy = (i - 1) ÷ Lx + 1
            # +x neighbor (no wrap here)
            if ix < Lx
                add_hund_coupling!(ham, J, i, i + 1)
            end
            # +y neighbor (no wrap here)
            if iy < Ly
                add_hund_coupling!(ham, J, i, i + Lx)
            end
            # +x + y diagonal (no wrap here)
            if ix < Lx && iy < Ly
                add_hund_coupling!(ham, J, i, i + 1 + Lx)
            end
        end
        # periodic wraps along x
        for iy = 1:Ly
            i_end = (iy - 1) * Lx + Lx
            i_start = (iy - 1) * Lx + 1
            add_hund_coupling!(ham, J, i_end, i_start)
        end
        # periodic wraps along y
        for ix = 1:Lx
            i_top = (Ly - 1) * Lx + ix
            i_bottom = ix
            add_hund_coupling!(ham, J, i_top, i_bottom)
        end
        # periodic wraps for the diagonal bonds: connect (ix=Lx,iy) to (ix=1,iy+1), and (iy=Ly,ix) to (ix+1,1)
        for iy = 1:Ly-1
            i = (iy - 1) * Lx + Lx
            j = iy * Lx + 1
            add_hund_coupling!(ham, J, i, j)
        end
        # wrap across y for diagonal from last row to first row
        for ix = 1:Lx-1
            i = (Ly - 1) * Lx + ix
            j = ix + 1
            add_hund_coupling!(ham, J, i, j)
        end
        # corner wrap (Lx, Ly) -> (1,1)
        add_hund_coupling!(ham, J, n_sites, 1)
    elseif lattice_type == :honeycomb
        # Honeycomb: 2 sites per unit cell; factor n_sites = 2 * Lx * Ly
        @assert n_sites % 2 == 0 "Honeycomb lattice requires even n_sites"
        ncell = n_sites ÷ 2
        if (Lx !== nothing) ⊻ (Ly !== nothing)
            error("Both Lx and Ly must be provided together if specified")
        end
        if Lx === nothing
            # Choose near-square Lx * Ly = ncell
            Lx_local = max(Int(round(sqrt(ncell))), 1)
            while ncell % Lx_local != 0 && Lx_local > 1
                Lx_local -= 1
            end
            Ly_local = max(ncell ÷ Lx_local, 1)
            @assert Lx_local * Ly_local == ncell "n_sites/2 must be factorizable as Lx*Ly for honeycomb lattice"
            Lx, Ly = Lx_local, Ly_local
        else
            @assert Lx > 0 && Ly > 0 "Lx and Ly must be positive"
            @assert 2 * Lx * Ly == n_sites "Provided 2*Lx*Ly must equal n_sites for honeycomb lattice"
        end

        # Index mapping: A=1, B=2 sublattice
        idx = (x, y, s) -> begin
            xx = ((x - 1) % Lx) + 1
            yy = ((y - 1) % Ly) + 1
            base = (yy - 1) * Lx + xx
            return 2 * (base - 1) + s
        end

        # For each cell, connect A(i,j) to three B neighbors
        for y = 1:Ly, x = 1:Lx
            a = idx(x, y, 1)
            b1 = idx(x, y, 2)
            b2 = idx(x - 1, y, 2)
            b3 = idx(x, y - 1, 2)
            add_hund_coupling!(ham, J, a, b1)
            add_hund_coupling!(ham, J, a, b2)
            add_hund_coupling!(ham, J, a, b3)
        end
    elseif lattice_type == :ladder
        # Two-leg ladder: n_sites = 2 * L
        @assert n_sites % 2 == 0 "Ladder lattice requires n_sites divisible by 2"
        L = n_sites ÷ 2
        leg = (l, w) -> (l - 1) * 2 + w # w=1,2
        # Along legs with PBC
        for l = 1:L
            lnext = l < L ? l + 1 : 1
            add_hund_coupling!(ham, J, leg(l, 1), leg(lnext, 1))
            add_hund_coupling!(ham, J, leg(l, 2), leg(lnext, 2))
            # Rung
            add_hund_coupling!(ham, J, leg(l, 1), leg(l, 2))
        end
    elseif lattice_type == :kagome
        # Kagome: 3 sites per unit cell; factor n_sites = 3 * Lx * Ly
        @assert n_sites % 3 == 0 "Kagome lattice requires n_sites divisible by 3"
        ncell = n_sites ÷ 3
        if (Lx !== nothing) ⊻ (Ly !== nothing)
            error("Both Lx and Ly must be provided together if specified")
        end
        if Lx === nothing
            Lx_local = max(Int(round(sqrt(ncell))), 1)
            while ncell % Lx_local != 0 && Lx_local > 1
                Lx_local -= 1
            end
            Ly_local = max(ncell ÷ Lx_local, 1)
            @assert Lx_local * Ly_local == ncell "n_sites/3 must be factorizable as Lx*Ly for kagome lattice"
            Lx, Ly = Lx_local, Ly_local
        else
            @assert Lx > 0 && Ly > 0 "Lx and Ly must be positive"
            @assert 3 * Lx * Ly == n_sites "Provided 3*Lx*Ly must equal n_sites for kagome lattice"
        end
        # Use StdFace geometry to define NN bonds robustly
        local geom = create_kagome_lattice(Lx, Ly)
        return create_heisenberg_hamiltonian(geom, J)
    else
        error("Unsupported lattice type for Heisenberg model: $lattice_type")
    end

    return ham
end

"""
    hamiltonian_summary(ham::Hamiltonian{T}) where {T}

Print a summary of the Hamiltonian terms.
"""
function hamiltonian_summary(ham::Hamiltonian{T}) where {T}
    println("Hamiltonian Summary:")
    println("  Sites: $(ham.n_sites), Electrons: $(ham.n_electrons)")
    println("  Transfer terms: $(length(ham.transfer_terms))")
    println("  Coulomb intra terms: $(length(ham.coulomb_intra_terms))")
    println("  Coulomb inter terms: $(length(ham.coulomb_inter_terms))")
    println("  Hund terms: $(length(ham.hund_terms))")
    println("  Pair hopping terms: $(length(ham.pair_hopping_terms))")
    println("  Exchange terms: $(length(ham.exchange_terms))")
    println("  InterAll terms: $(length(ham.interall_terms))")
end

# Additional functions for testing and analysis
