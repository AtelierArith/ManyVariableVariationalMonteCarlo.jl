"""
Integration of quantum projection into VMC calculations.

This module provides the integration between the C-compatible quantum projection
implementation and the main VMC calculation workflow.
"""

"""
    apply_quantum_projection_to_wavefunction!(sim::EnhancedVMCSimulation{T},
                                            electron_config::Vector{Int}) where T

Apply quantum projection to the current wavefunction configuration.
This modifies the wavefunction amplitude according to quantum number constraints.
"""
function apply_quantum_projection_to_wavefunction!(
    sim::EnhancedVMCSimulation{T},
    electron_config::Vector{Int}
) where {T}
    if sim.quantum_projection === nothing
        return one(T)  # No projection, return unity
    end

    qp = sim.quantum_projection
    if !qp.is_initialized
        @warn "Quantum projection not initialized, skipping projection"
        return one(T)
    end

    # For now, return unity as a placeholder for the full projection calculation
    # In a complete implementation, this would:
    # 1. Calculate the Pfaffian matrix for current configuration
    # 2. Apply quantum projection weights
    # 3. Return the projected amplitude

    return one(T)
end

"""
    calculate_quantum_projection_ratio(sim::EnhancedVMCSimulation{T},
                                     old_config::Vector{Int},
                                     new_config::Vector{Int}) where T

Calculate the ratio of quantum projection factors for configuration update.
This is used in Monte Carlo acceptance probability calculation.
"""
function calculate_quantum_projection_ratio(
    sim::EnhancedVMCSimulation{T},
    old_config::Vector{Int},
    new_config::Vector{Int}
) where {T}
    if sim.quantum_projection === nothing
        return one(T)  # No projection, return unity ratio
    end

    qp = sim.quantum_projection
    if !qp.is_initialized
        return one(T)
    end

    # Calculate projection factor for old configuration
    old_factor = apply_quantum_projection_to_wavefunction!(sim, old_config)

    # Calculate projection factor for new configuration
    new_factor = apply_quantum_projection_to_wavefunction!(sim, new_config)

    # Return ratio (avoid division by zero)
    if abs(old_factor) < 1e-15
        return zero(T)
    else
        return new_factor / old_factor
    end
end

"""
    update_quantum_projection_weights!(sim::EnhancedVMCSimulation{T}) where T

Update quantum projection weights during parameter optimization.
This should be called after updating variational parameters.
"""
function update_quantum_projection_weights!(sim::EnhancedVMCSimulation{T}) where {T}
    if sim.quantum_projection === nothing
        return
    end

    qp = sim.quantum_projection
    if !qp.is_initialized
        return
    end

    # Update optimization transformation parameters from current parameters
    if length(sim.parameters.opttrans) > 0 && length(qp.opt_trans) == length(sim.parameters.opttrans)
        qp.opt_trans .= sim.parameters.opttrans
        update_qp_weight!(qp)
    end
end

"""
    get_quantum_projection_info(sim::EnhancedVMCSimulation{T}) where T

Get information about the current quantum projection setup.
Returns a dictionary with projection parameters and status.
"""
function get_quantum_projection_info(sim::EnhancedVMCSimulation{T}) where {T}
    if sim.quantum_projection === nothing
        return Dict{String,Any}(
            "enabled" => false,
            "reason" => "No quantum projection initialized"
        )
    end

    qp = sim.quantum_projection

    return Dict{String,Any}(
        "enabled" => qp.is_initialized,
        "n_sp_gauss_leg" => qp.n_sp_gauss_leg,
        "n_mp_trans" => qp.n_mp_trans,
        "n_sp_stot" => qp.n_sp_stot,
        "n_qp_fix" => qp.n_qp_fix,
        "n_qp_full" => qp.n_qp_full,
        "n_opt_trans" => qp.n_opt_trans,
        "has_spin_projection" => qp.n_sp_gauss_leg > 1,
        "has_momentum_projection" => qp.n_mp_trans > 1,
        "has_optimization_trans" => qp.n_opt_trans > 1
    )
end

"""
    print_quantum_projection_summary(sim::EnhancedVMCSimulation{T}) where T

Print a summary of the quantum projection setup.
"""
function print_quantum_projection_summary(sim::EnhancedVMCSimulation{T}) where {T}
    info = get_quantum_projection_info(sim)

    println("######  Quantum Projection Summary  ######")
    if info["enabled"]
        println("Quantum projection: ENABLED")
        println("  Spin projection points: ", info["n_sp_gauss_leg"])
        println("  Momentum projections: ", info["n_mp_trans"])
        println("  Total spin: ", info["n_sp_stot"])
        println("  Fixed QP dimension: ", info["n_qp_fix"])
        println("  Full QP dimension: ", info["n_qp_full"])
        println("  Optimization transformations: ", info["n_opt_trans"])

        if info["has_spin_projection"]
            println("  → Spin projection with Gauss-Legendre quadrature")
        end
        if info["has_momentum_projection"]
            println("  → Momentum projection with translational symmetry")
        end
        if info["has_optimization_trans"]
            println("  → Optimization transformation enabled")
        end
    else
        println("Quantum projection: DISABLED")
        println("  Reason: ", info["reason"])
    end
    println("##########################################")
    println()
end
