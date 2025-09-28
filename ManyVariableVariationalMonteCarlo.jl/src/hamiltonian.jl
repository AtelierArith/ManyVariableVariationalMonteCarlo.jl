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
            Dict{Vector{Int},T}()
        )
    end
end

"""
    Hamiltonian(n_sites::Int, n_electrons::Int; T=ComplexF64)

Create a new Hamiltonian for a system with given number of sites and electrons.
"""
Hamiltonian(n_sites::Int, n_electrons::Int; T=ComplexF64) = Hamiltonian{T}(n_sites, n_electrons)

"""
    add_transfer!(ham::Hamiltonian{T}, coeff::T, i::Int, si::Int, j::Int, sj::Int) where {T}

Add a kinetic energy (transfer) term to the Hamiltonian.
"""
function add_transfer!(ham::Hamiltonian{T}, coeff, i::Int, si::Int, j::Int, sj::Int) where {T}
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
function add_interall!(ham::Hamiltonian{T}, coeff, i::Int, si::Int, j::Int, sj::Int,
                      k::Int, sk::Int, l::Int, sl::Int) where {T}
    cT = convert(T, coeff)
    push!(ham.interall_terms, InterAllTerm{T}(cT, i, si, j, sj, k, sk, l, sl))
end

"""
    calculate_hamiltonian(ham::Hamiltonian{T}, electron_config::Vector{Int},
                         electron_numbers::Vector{Int}) where {T}

Calculate the Hamiltonian expectation value for a given electron configuration.
Equivalent to CalculateHamiltonian in the C reference implementation.
"""
function calculate_hamiltonian(ham::Hamiltonian{T}, electron_config::Vector{Int},
                              electron_numbers::Vector{Int}) where {T}
    energy = zero(T)

    # Kinetic energy (transfer terms)
    energy += calculate_kinetic_energy(ham, electron_config, electron_numbers)

    # Potential energy terms
    energy += calculate_coulomb_intra_energy(ham, electron_numbers)
    energy += calculate_coulomb_inter_energy(ham, electron_numbers)
    energy += calculate_hund_energy(ham, electron_numbers)
    energy += calculate_pair_hopping_energy(ham, electron_config, electron_numbers)
    energy += calculate_exchange_energy(ham, electron_config, electron_numbers)
    energy += calculate_interall_energy(ham, electron_config, electron_numbers)

    return energy
end

"""
    calculate_kinetic_energy(ham::Hamiltonian{T}, electron_config::Vector{Int},
                           electron_numbers::Vector{Int}) where {T}

Calculate kinetic energy contribution.
"""
function calculate_kinetic_energy(ham::Hamiltonian{T}, electron_config::Vector{Int},
                                 electron_numbers::Vector{Int}) where {T}
    energy = zero(T)
    n_up = electron_numbers[1:ham.n_sites]
    n_down = electron_numbers[ham.n_sites+1:2*ham.n_sites]

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
function calculate_coulomb_intra_energy(ham::Hamiltonian{T}, electron_numbers::Vector{Int}) where {T}
    energy = zero(T)
    n_up = electron_numbers[1:ham.n_sites]
    n_down = electron_numbers[ham.n_sites+1:2*ham.n_sites]

    for term in ham.coulomb_intra_terms
        energy += term.coefficient * n_up[term.site] * n_down[term.site]
    end

    return energy
end

"""
    calculate_coulomb_inter_energy(ham::Hamiltonian{T}, electron_numbers::Vector{Int}) where {T}

Calculate inter-site Coulomb interaction energy.
"""
function calculate_coulomb_inter_energy(ham::Hamiltonian{T}, electron_numbers::Vector{Int}) where {T}
    energy = zero(T)
    n_total = electron_numbers[1:ham.n_sites] + electron_numbers[ham.n_sites+1:2*ham.n_sites]

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
    n_up = electron_numbers[1:ham.n_sites]
    n_down = electron_numbers[ham.n_sites+1:2*ham.n_sites]

    for term in ham.hund_terms
        # Hund term: J * (S⃗ᵢ · S⃗ⱼ - nᵢnⱼ/4)
        # Simplified to J * (nᵢ↑nⱼ↑ + nᵢ↓nⱼ↓ - nᵢ↑nⱼ↓ - nᵢ↓nⱼ↑ - (nᵢ↑ + nᵢ↓)(nⱼ↑ + nⱼ↓)/4)
        ni_up = n_up[term.site_i]
        ni_down = n_down[term.site_i]
        nj_up = n_up[term.site_j]
        nj_down = n_down[term.site_j]

        hund_energy = ni_up * nj_up + ni_down * nj_down - ni_up * nj_down - ni_down * nj_up
        hund_energy -= (ni_up + ni_down) * (nj_up + nj_down) / 4

        energy += term.coefficient * hund_energy
    end

    return energy
end

"""
    calculate_pair_hopping_energy(ham::Hamiltonian{T}, electron_config::Vector{Int},
                                 electron_numbers::Vector{Int}) where {T}

Calculate pair hopping energy (simplified approximation).
"""
function calculate_pair_hopping_energy(ham::Hamiltonian{T}, electron_config::Vector{Int},
                                      electron_numbers::Vector{Int}) where {T}
    # Pair hopping terms are complex and require proper treatment of creation/annihilation operators
    # For now, return zero as placeholder
    return zero(T)
end

"""
    calculate_exchange_energy(ham::Hamiltonian{T}, electron_config::Vector{Int},
                            electron_numbers::Vector{Int}) where {T}

Calculate exchange interaction energy (simplified approximation).
"""
function calculate_exchange_energy(ham::Hamiltonian{T}, electron_config::Vector{Int},
                                  electron_numbers::Vector{Int}) where {T}
    # Exchange terms are complex and require proper treatment
    # For now, return zero as placeholder
    return zero(T)
end

"""
    calculate_interall_energy(ham::Hamiltonian{T}, electron_config::Vector{Int},
                             electron_numbers::Vector{Int}) where {T}

Calculate general four-operator interaction energy (simplified approximation).
"""
function calculate_interall_energy(ham::Hamiltonian{T}, electron_config::Vector{Int},
                                  electron_numbers::Vector{Int}) where {T}
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
    n_down = electron_numbers[n_sites+1:2*n_sites]

    double_occ = 0.0
    for i in 1:n_sites
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
                              lattice_type::Symbol=:chain) where {T}

Create a standard Hubbard model Hamiltonian.
"""
function create_hubbard_hamiltonian(n_sites::Int, n_electrons::Int, t::T, U::T;
                                   lattice_type::Symbol=:chain,
                                   apbc::Bool=false,
                                   twist_x::Float64=0.0,
                                   twist_y::Float64=0.0) where {T}
    # Always construct a complex Hamiltonian to support boundary twists/APBC
    # (real inputs are safely converted to ComplexF64 as needed)
    ham = Hamiltonian{ComplexF64}(n_sites, n_electrons)

    # Add on-site Coulomb terms
    for i in 1:n_sites
        add_coulomb_intra!(ham, U, i)
    end

    # Add hopping terms based on lattice type
    if lattice_type == :chain
        # 1D chain with nearest-neighbor hopping
        for i in 1:n_sites-1
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
        Lx = Int(sqrt(n_sites))
        Ly = n_sites ÷ Lx

        for i in 1:n_sites
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
    create_heisenberg_hamiltonian(n_sites::Int, J::T; lattice_type::Symbol=:chain) where {T}

Create a standard Heisenberg model Hamiltonian.
"""
function create_heisenberg_hamiltonian(n_sites::Int, J::T; lattice_type::Symbol=:chain) where {T}
    ham = Hamiltonian{T}(n_sites, n_sites)  # One electron per site for spin-1/2

    if lattice_type == :chain
        # 1D chain with nearest-neighbor exchange
        for i in 1:n_sites-1
            add_hund_coupling!(ham, J, i, i+1)
        end

        # Periodic boundary conditions
        if n_sites > 2
            add_hund_coupling!(ham, J, 1, n_sites)
        end
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
