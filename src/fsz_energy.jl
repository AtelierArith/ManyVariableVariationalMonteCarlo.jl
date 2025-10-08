"""
FSZ Energy Calculation

Energy calculation for FSZ (Fixed Spin Zone) sampling.
Computes local energy for spin systems (Heisenberg, Hubbard, etc.).

Based on mVMC C implementation:
- calham.c: Hamiltonian calculations
- vmccal_fsz.c: FSZ-specific calculations
"""

using LinearAlgebra

"""
    FSZEnergyCalculator{T}

Calculator for local energy in FSZ sampling.
"""
struct FSZEnergyCalculator{T<:Number}
    n_sites::Int
    n_elec::Int

    # Hamiltonian parameters
    transfer::Vector{Tuple{Int,Int,T}}      # (i, j, t_ij)
    coulomb_intra::Vector{Tuple{Int,T}}     # (i, U_i)
    coulomb_inter::Vector{Tuple{Int,Int,T}} # (i, j, V_ij)
    exchange::Vector{Tuple{Int,Int,T}}      # (i, j, J_ij)
    pairhop::Vector{Tuple{Int,Int,T}}       # (i, j, P_ij)

    # For spin systems
    is_spin_system::Bool

    function FSZEnergyCalculator{T}(
        n_sites::Int,
        n_elec::Int;
        is_spin_system::Bool = false,
    ) where {T}
        new{T}(
            n_sites,
            n_elec,
            Tuple{Int,Int,T}[],
            Tuple{Int,T}[],
            Tuple{Int,Int,T}[],
            Tuple{Int,Int,T}[],
            Tuple{Int,Int,T}[],
            is_spin_system,
        )
    end
end

"""
    add_transfer!(calc::FSZEnergyCalculator{T}, i::Int, j::Int, t::T) where T

Add transfer (hopping) term: t * c†_i c_j + h.c.

C実装参考: calham.c 1行目から522行目まで
"""
function add_transfer!(calc::FSZEnergyCalculator{T}, i::Int, j::Int, t::T) where {T}
    push!(calc.transfer, (i, j, t))
    return nothing
end

"""
    add_coulomb_intra!(calc::FSZEnergyCalculator{T}, i::Int, U::T) where T

Add intra-site Coulomb interaction: U * n_i↑ * n_i↓
"""
function add_coulomb_intra!(calc::FSZEnergyCalculator{T}, i::Int, U::T) where {T}
    push!(calc.coulomb_intra, (i, U))
    return nothing
end

"""
    add_coulomb_inter!(calc::FSZEnergyCalculator{T}, i::Int, j::Int, V::T) where T

Add inter-site Coulomb interaction: V * n_i * n_j
"""
function add_coulomb_inter!(calc::FSZEnergyCalculator{T}, i::Int, j::Int, V::T) where {T}
    push!(calc.coulomb_inter, (i, j, V))
    return nothing
end

"""
    add_exchange!(calc::FSZEnergyCalculator{T}, i::Int, j::Int, J::T) where T

Add exchange interaction: J * (S_i · S_j - n_i*n_j/4)
For spin systems: J * (S_i · S_j)
"""
function add_exchange!(calc::FSZEnergyCalculator{T}, i::Int, j::Int, J::T) where {T}
    push!(calc.exchange, (i, j, J))
    return nothing
end

"""
    calculate_local_energy(
        calc::FSZEnergyCalculator{T},
        state::FSZSamplerState{T}
    ) where T

Calculate local energy for current electron configuration.

E_local = <ψ|H|ψ_current> / <ψ|ψ_current>
        = Σ_terms H_term

For spin systems (Heisenberg):
    H = Σ_{<i,j>} J_ij S_i · S_j
"""
function calculate_local_energy(
    calc::FSZEnergyCalculator{T},
    state::FSZSamplerState{T},
) where {T}
    energy = zero(T)

    if calc.is_spin_system
        # Heisenberg-type model
        energy += calculate_exchange_energy(calc, state)
    else
        # Hubbard-type model
        energy += calculate_transfer_energy(calc, state)
        energy += calculate_coulomb_energy(calc, state)
        energy += calculate_exchange_energy(calc, state)
    end

    return energy
end

"""
    calculate_exchange_energy(
        calc::FSZEnergyCalculator{T},
        state::FSZSamplerState{T}
    ) where T

Calculate exchange interaction energy.

For spin systems: E = Σ J_ij * S_i · S_j
where S_i · S_j = (S^z_i * S^z_j) + 0.5 * (S^+_i * S^-_j + S^-_i * S^+_j)
"""
function calculate_exchange_energy(
    calc::FSZEnergyCalculator{T},
    state::FSZSamplerState{T},
) where {T}
    energy = zero(T)
    n_sites = calc.n_sites

    for (i, j, J) in calc.exchange
        # Get occupation at sites i and j
        # For spin system: sites have one electron with spin up or down

        # Site i
        i_up = state.ele_cfg[i+1]  # electron index at site i, up spin
        i_down = state.ele_cfg[i+1 + n_sites]  # electron index at site i, down spin

        # Site j
        j_up = state.ele_cfg[j+1]
        j_down = state.ele_cfg[j+1 + n_sites]

        # Calculate S^z contributions
        # S^z_i = (n_i↑ - n_i↓) / 2
        sz_i = ((i_up >= 0 ? 1 : 0) - (i_down >= 0 ? 1 : 0)) / 2.0
        sz_j = ((j_up >= 0 ? 1 : 0) - (j_down >= 0 ? 1 : 0)) / 2.0

        # S^z_i * S^z_j term
        energy += J * sz_i * sz_j

        # S^+ S^- terms (flip-flop)
        # S^+_i S^-_j: site i has down, site j has up -> flip both
        # S^-_i S^+_j: site i has up, site j has down -> flip both

        # This would require calculating off-diagonal matrix elements
        # For simplified version, use only diagonal term
        # Full implementation would use ratio calculations

        if i_down >= 0 && j_up >= 0
            # Can flip: i down->up, j up->down
            # Contribution: J/2 * ratio
            # For now, add average contribution
            energy += J * 0.25  # Simplified
        end

        if i_up >= 0 && j_down >= 0
            # Can flip: i up->down, j down->up
            energy += J * 0.25  # Simplified
        end
    end

    return energy
end

"""
    calculate_transfer_energy(
        calc::FSZEnergyCalculator{T},
        state::FSZSamplerState{T}
    ) where {T

Calculate kinetic (transfer/hopping) energy.
"""
function calculate_transfer_energy(
    calc::FSZEnergyCalculator{T},
    state::FSZSamplerState{T},
) where {T}
    # Transfer energy requires off-diagonal matrix elements
    # Would need to calculate ratios for hopping moves
    # Simplified version returns zero
    return zero(T)
end

"""
    calculate_coulomb_energy(
        calc::FSZEnergyCalculator{T},
        state::FSZSamplerState{T}
    ) where {T}

Calculate Coulomb interaction energy (diagonal contribution).
"""
function calculate_coulomb_energy(
    calc::FSZEnergyCalculator{T},
    state::FSZSamplerState{T},
) where {T}
    energy = zero(T)
    n_sites = calc.n_sites

    # Intra-site Coulomb
    for (i, U) in calc.coulomb_intra
        i_up = state.ele_cfg[i+1]
        i_down = state.ele_cfg[i+1 + n_sites]

        if i_up >= 0 && i_down >= 0
            # Double occupancy
            energy += U
        end
    end

    # Inter-site Coulomb
    for (i, j, V) in calc.coulomb_inter
        i_occ = (state.ele_cfg[i+1] >= 0 || state.ele_cfg[i+1 + n_sites] >= 0) ? 1 : 0
        j_occ = (state.ele_cfg[j+1] >= 0 || state.ele_cfg[j+1 + n_sites] >= 0) ? 1 : 0

        energy += V * i_occ * j_occ
    end

    return energy
end

"""
    create_heisenberg_chain_energy_calculator(
        n_sites::Int,
        J::Float64
    ) -> FSZEnergyCalculator{Float64}

Create energy calculator for Heisenberg chain.

H = Σ_{<i,j>} J * S_i · S_j
"""
function create_heisenberg_chain_energy_calculator(
    n_sites::Int,
    J::Float64,
)
    calc = FSZEnergyCalculator{Float64}(n_sites, n_sites; is_spin_system=true)

    # Add nearest-neighbor exchange interactions
    for i in 0:(n_sites-1)
        j = (i + 1) % n_sites  # Periodic boundary conditions
        add_exchange!(calc, i, j, J)
    end

    return calc
end

"""
    calculate_energy_and_variance(
        calc::FSZEnergyCalculator{T},
        samples::Vector{FSZSamplerState{T}}
    ) where T

Calculate average energy and variance from FSZ samples.

# Returns
- `energy_mean`: Average energy
- `energy_variance`: Variance of energy
- `energy_std`: Standard deviation
"""
function calculate_energy_and_variance(
    calc::FSZEnergyCalculator{T},
    samples::Vector{FSZSamplerState{T}},
) where {T}
    n_samples = length(samples)

    if n_samples == 0
        return (zero(T), zero(T), zero(T))
    end

    energies = [calculate_local_energy(calc, sample) for sample in samples]

    energy_mean = sum(energies) / n_samples
    energy_variance = sum((e - energy_mean)^2 for e in energies) / n_samples
    energy_std = sqrt(energy_variance)

    return (energy_mean, energy_variance, energy_std)
end

"""
    print_energy_info(calc::FSZEnergyCalculator)

Print information about energy calculator.
"""
function print_energy_info(calc::FSZEnergyCalculator)
    println("FSZ Energy Calculator:")
    println("  n_sites: $(calc.n_sites)")
    println("  n_elec: $(calc.n_elec)")
    println("  is_spin_system: $(calc.is_spin_system)")
    println("  n_transfer: $(length(calc.transfer))")
    println("  n_coulomb_intra: $(length(calc.coulomb_intra))")
    println("  n_coulomb_inter: $(length(calc.coulomb_inter))")
    println("  n_exchange: $(length(calc.exchange))")
end
