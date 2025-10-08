"""
Observable Measurement System for ManyVariableVariationalMonteCarlo.jl

Implements measurement of various observables including:
- Energy calculations
- Correlation functions
- Custom observables
- Statistical analysis
- Measurement accumulation

Ported from calham*.c and average.c in the C reference implementation.
"""

using LinearAlgebra
using Statistics
using Random

"""
    ObservableType

Enumeration of different observable types.
"""
@enum ObservableType begin
    ENERGY
    KINETIC_ENERGY
    POTENTIAL_ENERGY
    SPIN_CORRELATION
    DENSITY_CORRELATION
    MOMENTUM_DISTRIBUTION
    CUSTOM_OBSERVABLE
end

"""
    ObservableMeasurement{T}

Represents a single observable measurement.
"""
mutable struct ObservableMeasurement{T<:Union{Float64,ComplexF64}}
    observable_type::ObservableType
    value::T
    weight::Float64
    measurement_time::Float64

    function ObservableMeasurement{T}(
        obs_type::ObservableType,
        value::T,
        weight::Float64 = 1.0,
    ) where {T}
        new{T}(obs_type, value, weight, 0.0)
    end
end

"""
    ObservableAccumulator{T}

Accumulates measurements for statistical analysis.
"""
mutable struct ObservableAccumulator{T<:Union{Float64,ComplexF64}}
    # Accumulated values
    sum_values::T
    sum_squared_values::T
    sum_weights::Float64
    sum_weighted_values::T
    sum_weighted_squared_values::T

    # Statistics
    mean_value::T
    variance::T
    standard_error::T
    n_measurements::Int

    # Measurement history
    measurements::Vector{ObservableMeasurement{T}}
    max_history::Int

    function ObservableAccumulator{T}(max_history::Int = 1000) where {T}
        new{T}(
            zero(T),
            zero(T),
            0.0,
            zero(T),
            zero(T),
            zero(T),
            zero(T),
            zero(T),
            0,
            ObservableMeasurement{T}[],
            max_history,
        )
    end
end

"""
    add_measurement!(accumulator::ObservableAccumulator{T}, measurement::ObservableMeasurement{T}) where T

Add a measurement to the accumulator.

C実装参考: calham.c 1行目から522行目まで、average.c 1行目から334行目まで
"""
function add_measurement!(
    accumulator::ObservableAccumulator{T},
    measurement::ObservableMeasurement{T},
) where {T}
    # Update accumulators
    accumulator.sum_values += measurement.value
    accumulator.sum_squared_values += measurement.value^2
    accumulator.sum_weights += measurement.weight
    accumulator.sum_weighted_values += measurement.weight * measurement.value
    accumulator.sum_weighted_squared_values += measurement.weight * measurement.value^2
    accumulator.n_measurements += 1

    # Add to history
    push!(accumulator.measurements, measurement)
    if length(accumulator.measurements) > accumulator.max_history
        popfirst!(accumulator.measurements)
    end

    # Update statistics
    update_statistics!(accumulator)
end

"""
    update_statistics!(accumulator::ObservableAccumulator{T}) where T

Update statistical quantities.
"""
function update_statistics!(accumulator::ObservableAccumulator{T}) where {T}
    if accumulator.n_measurements == 0
        return
    end

    # Calculate mean
    if accumulator.sum_weights > 0
        accumulator.mean_value = accumulator.sum_weighted_values / accumulator.sum_weights
    else
        accumulator.mean_value = accumulator.sum_values / accumulator.n_measurements
    end

    # Calculate variance
    if accumulator.n_measurements > 1
        if accumulator.sum_weights > 0
            # Weighted variance
            accumulator.variance =
                (accumulator.sum_weighted_squared_values / accumulator.sum_weights) -
                (accumulator.mean_value)^2
        else
            # Unweighted variance
            accumulator.variance =
                (accumulator.sum_squared_values / accumulator.n_measurements) -
                (accumulator.mean_value)^2
        end

        # Standard error
        accumulator.standard_error =
            sqrt(real(accumulator.variance) / (accumulator.n_measurements - 1))
    else
        accumulator.variance = zero(T)
        accumulator.standard_error = zero(T)
    end
end

"""
    EnergyCalculator{T}

Calculates various energy components.
"""
mutable struct EnergyCalculator{T<:Union{Float64,ComplexF64}}
    # System parameters
    n_site::Int
    n_elec::Int

    # Hamiltonian parameters
    hopping_matrix::Matrix{T}
    interaction_matrix::Matrix{T}
    external_potential::Vector{T}

    # Working arrays
    energy_buffer::Vector{T}
    gradient_buffer::Vector{T}

    # Statistics
    total_calculations::Int
    calculation_time::Float64

    function EnergyCalculator{T}(n_site::Int, n_elec::Int) where {T}
        hopping_matrix = zeros(T, n_site, n_site)
        interaction_matrix = zeros(T, n_site, n_site)
        external_potential = zeros(T, n_site)
        energy_buffer = Vector{T}(undef, n_site)
        gradient_buffer = Vector{T}(undef, n_site)

        new{T}(
            n_site,
            n_elec,
            hopping_matrix,
            interaction_matrix,
            external_potential,
            energy_buffer,
            gradient_buffer,
            0,
            0.0,
        )
    end
end

"""
    calculate_kinetic_energy(calc::EnergyCalculator{T}, ele_idx::Vector{Int},
                            ele_cfg::Vector{Int}) where T

Calculate kinetic energy for given electron configuration.
"""
function calculate_kinetic_energy(
    calc::EnergyCalculator{T},
    ele_idx::Vector{Int},
    ele_cfg::Vector{Int},
) where {T}
    kinetic_energy = zero(T)

    # Sum over all electron pairs
    for i = 1:length(ele_idx)
        for j = 1:length(ele_idx)
            if i != j
                site_i = ele_idx[i]
                site_j = ele_idx[j]
                kinetic_energy += calc.hopping_matrix[site_i, site_j]
            end
        end
    end

    calc.total_calculations += 1
    return kinetic_energy
end

"""
    calculate_potential_energy(calc::EnergyCalculator{T}, ele_idx::Vector{Int},
                              ele_cfg::Vector{Int}) where T

Calculate potential energy for given electron configuration.
"""
function calculate_potential_energy(
    calc::EnergyCalculator{T},
    ele_idx::Vector{Int},
    ele_cfg::Vector{Int},
) where {T}
    potential_energy = zero(T)

    # Interaction energy
    for i = 1:length(ele_idx)
        for j = 1:length(ele_idx)
            if i != j
                site_i = ele_idx[i]
                site_j = ele_idx[j]
                potential_energy += calc.interaction_matrix[site_i, site_j]
            end
        end
    end

    # External potential energy
    for i = 1:length(ele_idx)
        site_i = ele_idx[i]
        potential_energy += calc.external_potential[site_i]
    end

    calc.total_calculations += 1
    return potential_energy
end

"""
    calculate_total_energy(calc::EnergyCalculator{T}, ele_idx::Vector{Int},
                          ele_cfg::Vector{Int}) where T

Calculate total energy for given electron configuration.
"""
function calculate_total_energy(
    calc::EnergyCalculator{T},
    ele_idx::Vector{Int},
    ele_cfg::Vector{Int},
) where {T}
    kinetic = calculate_kinetic_energy(calc, ele_idx, ele_cfg)
    potential = calculate_potential_energy(calc, ele_idx, ele_cfg)
    return kinetic + potential
end

"""
    CorrelationFunction{T}

Calculates correlation functions.
"""
mutable struct CorrelationFunction{T<:Union{Float64,ComplexF64}}
    # System parameters
    n_site::Int
    n_elec::Int

    # Correlation parameters
    max_distance::Int
    correlation_type::String

    # Working arrays
    correlation_buffer::Vector{T}
    distance_buffer::Vector{Int}

    # Statistics
    total_calculations::Int
    calculation_time::Float64

    function CorrelationFunction{T}(
        n_site::Int,
        n_elec::Int;
        max_distance::Int = 5,
        correlation_type::String = "spin",
    ) where {T}
        correlation_buffer = Vector{T}(undef, max_distance + 1)
        distance_buffer = Vector{Int}(undef, n_site * n_site)

        new{T}(
            n_site,
            n_elec,
            max_distance,
            correlation_type,
            correlation_buffer,
            distance_buffer,
            0,
            0.0,
        )
    end
end

"""
    calculate_spin_correlation(corr::CorrelationFunction{T}, ele_idx::Vector{Int},
                              ele_cfg::Vector{Int}) where T

Calculate spin correlation function.
"""
function calculate_spin_correlation(
    corr::CorrelationFunction{T},
    ele_idx::Vector{Int},
    ele_cfg::Vector{Int},
) where {T}
    # Initialize correlation function
    fill!(corr.correlation_buffer, zero(T))

    # Calculate correlations for each distance
    for d = 0:corr.max_distance
        correlation_sum = zero(T)
        count = 0

        for i = 1:length(ele_idx)
            for j = 1:length(ele_idx)
                if i != j
                    site_i = ele_idx[i]
                    site_j = ele_idx[j]
                    distance = abs(site_i - site_j)

                    if distance == d
                        # Simplified spin correlation (in real implementation, this would depend on spin configuration)
                        spin_correlation = (i == j) ? 1.0 : -1.0
                        correlation_sum += T(spin_correlation)
                        count += 1
                    end
                end
            end
        end

        if count > 0
            corr.correlation_buffer[d+1] = correlation_sum / count
        end
    end

    corr.total_calculations += 1
    return copy(corr.correlation_buffer)
end

"""
    calculate_density_correlation(corr::CorrelationFunction{T}, ele_idx::Vector{Int},
                                 ele_cfg::Vector{Int}) where T

Calculate density correlation function.
"""
function calculate_density_correlation(
    corr::CorrelationFunction{T},
    ele_idx::Vector{Int},
    ele_cfg::Vector{Int},
) where {T}
    # Initialize correlation function
    fill!(corr.correlation_buffer, zero(T))

    # Calculate correlations for each distance
    for d = 0:corr.max_distance
        correlation_sum = zero(T)
        count = 0

        for i = 1:length(ele_idx)
            for j = 1:length(ele_idx)
                if i != j
                    site_i = ele_idx[i]
                    site_j = ele_idx[j]
                    distance = abs(site_i - site_j)

                    if distance == d
                        # Density correlation (simplified)
                        density_correlation = ele_cfg[site_i] * ele_cfg[site_j]
                        correlation_sum += T(density_correlation)
                        count += 1
                    end
                end
            end
        end

        if count > 0
            corr.correlation_buffer[d+1] = correlation_sum / count
        end
    end

    corr.total_calculations += 1
    return copy(corr.correlation_buffer)
end

"""
    ObservableManager{T}

Manages multiple observables and their measurements.
"""
mutable struct ObservableManager{T<:Union{Float64,ComplexF64}}
    # Observables
    energy_calculator::EnergyCalculator{T}
    spin_correlation::CorrelationFunction{T}
    density_correlation::CorrelationFunction{T}

    # Accumulators
    energy_accumulator::ObservableAccumulator{T}
    kinetic_energy_accumulator::ObservableAccumulator{T}
    potential_energy_accumulator::ObservableAccumulator{T}
    spin_correlation_accumulator::ObservableAccumulator{T}
    density_correlation_accumulator::ObservableAccumulator{T}

    # Custom observables
    custom_observables::Dict{String,ObservableAccumulator{T}}

    # Statistics
    total_measurements::Int
    measurement_time::Float64

    function ObservableManager{T}(n_site::Int, n_elec::Int) where {T}
        energy_calculator = EnergyCalculator{T}(n_site, n_elec)
        spin_correlation = CorrelationFunction{T}(n_site, n_elec, correlation_type = "spin")
        density_correlation =
            CorrelationFunction{T}(n_site, n_elec, correlation_type = "density")

        energy_accumulator = ObservableAccumulator{T}()
        kinetic_energy_accumulator = ObservableAccumulator{T}()
        potential_energy_accumulator = ObservableAccumulator{T}()
        spin_correlation_accumulator = ObservableAccumulator{T}()
        density_correlation_accumulator = ObservableAccumulator{T}()

        custom_observables = Dict{String,ObservableAccumulator{T}}()

        new{T}(
            energy_calculator,
            spin_correlation,
            density_correlation,
            energy_accumulator,
            kinetic_energy_accumulator,
            potential_energy_accumulator,
            spin_correlation_accumulator,
            density_correlation_accumulator,
            custom_observables,
            0,
            0.0,
        )
    end
end

"""
    measure_observables!(manager::ObservableManager{T}, ele_idx::Vector{Int},
                        ele_cfg::Vector{Int}, weight::Float64 = 1.0) where T

Measure all observables for given electron configuration.
"""
function measure_observables!(
    manager::ObservableManager{T},
    ele_idx::Vector{Int},
    ele_cfg::Vector{Int},
    weight::Float64 = 1.0,
) where {T}
    # Measure energy
    total_energy = calculate_total_energy(manager.energy_calculator, ele_idx, ele_cfg)
    energy_measurement = ObservableMeasurement{T}(ENERGY, total_energy, weight)
    add_measurement!(manager.energy_accumulator, energy_measurement)

    # Measure kinetic energy
    kinetic_energy = calculate_kinetic_energy(manager.energy_calculator, ele_idx, ele_cfg)
    kinetic_measurement = ObservableMeasurement{T}(KINETIC_ENERGY, kinetic_energy, weight)
    add_measurement!(manager.kinetic_energy_accumulator, kinetic_measurement)

    # Measure potential energy
    potential_energy =
        calculate_potential_energy(manager.energy_calculator, ele_idx, ele_cfg)
    potential_measurement =
        ObservableMeasurement{T}(POTENTIAL_ENERGY, potential_energy, weight)
    add_measurement!(manager.potential_energy_accumulator, potential_measurement)

    # Measure spin correlation
    spin_corr = calculate_spin_correlation(manager.spin_correlation, ele_idx, ele_cfg)
    for (d, corr_val) in enumerate(spin_corr)
        spin_corr_measurement = ObservableMeasurement{T}(SPIN_CORRELATION, corr_val, weight)
        add_measurement!(manager.spin_correlation_accumulator, spin_corr_measurement)
    end

    # Measure density correlation
    density_corr =
        calculate_density_correlation(manager.density_correlation, ele_idx, ele_cfg)
    for (d, corr_val) in enumerate(density_corr)
        density_corr_measurement =
            ObservableMeasurement{T}(DENSITY_CORRELATION, corr_val, weight)
        add_measurement!(manager.density_correlation_accumulator, density_corr_measurement)
    end

    manager.total_measurements += 1
end

"""
    add_custom_observable!(manager::ObservableManager{T}, name::String) where T

Add a custom observable to the manager.
"""
function add_custom_observable!(manager::ObservableManager{T}, name::String) where {T}
    if !haskey(manager.custom_observables, name)
        manager.custom_observables[name] = ObservableAccumulator{T}()
    end
end

"""
    measure_custom_observable!(manager::ObservableManager{T}, name::String,
                              value::T, weight::Float64 = 1.0) where T

Measure a custom observable.
"""
function measure_custom_observable!(
    manager::ObservableManager{T},
    name::String,
    value::T,
    weight::Float64 = 1.0,
) where {T}
    if haskey(manager.custom_observables, name)
        measurement = ObservableMeasurement{T}(CUSTOM_OBSERVABLE, value, weight)
        add_measurement!(manager.custom_observables[name], measurement)
    else
        throw(
            ArgumentError(
                "Custom observable '$name' not found. Add it first with add_custom_observable!.",
            ),
        )
    end
end

"""
    get_observable_statistics(manager::ObservableManager{T}, observable_name::String) where T

Get statistics for a specific observable.
"""
function get_observable_statistics(
    manager::ObservableManager{T},
    observable_name::String,
) where {T}
    if observable_name == "energy"
        return (
            mean = manager.energy_accumulator.mean_value,
            variance = manager.energy_accumulator.variance,
            standard_error = manager.energy_accumulator.standard_error,
            n_measurements = manager.energy_accumulator.n_measurements,
        )
    elseif observable_name == "kinetic_energy"
        return (
            mean = manager.kinetic_energy_accumulator.mean_value,
            variance = manager.kinetic_energy_accumulator.variance,
            standard_error = manager.kinetic_energy_accumulator.standard_error,
            n_measurements = manager.kinetic_energy_accumulator.n_measurements,
        )
    elseif observable_name == "potential_energy"
        return (
            mean = manager.potential_energy_accumulator.mean_value,
            variance = manager.potential_energy_accumulator.variance,
            standard_error = manager.potential_energy_accumulator.standard_error,
            n_measurements = manager.potential_energy_accumulator.n_measurements,
        )
    elseif observable_name == "spin_correlation"
        return (
            mean = manager.spin_correlation_accumulator.mean_value,
            variance = manager.spin_correlation_accumulator.variance,
            standard_error = manager.spin_correlation_accumulator.standard_error,
            n_measurements = manager.spin_correlation_accumulator.n_measurements,
        )
    elseif observable_name == "density_correlation"
        return (
            mean = manager.density_correlation_accumulator.mean_value,
            variance = manager.density_correlation_accumulator.variance,
            standard_error = manager.density_correlation_accumulator.standard_error,
            n_measurements = manager.density_correlation_accumulator.n_measurements,
        )
    elseif haskey(manager.custom_observables, observable_name)
        acc = manager.custom_observables[observable_name]
        return (
            mean = acc.mean_value,
            variance = acc.variance,
            standard_error = acc.standard_error,
            n_measurements = acc.n_measurements,
        )
    else
        throw(ArgumentError("Observable '$observable_name' not found."))
    end
end

"""
    reset_observable_statistics!(manager::ObservableManager{T}) where T

Reset all observable statistics.
"""
function reset_observable_statistics!(manager::ObservableManager{T}) where {T}
    # Reset accumulators
    manager.energy_accumulator = ObservableAccumulator{T}()
    manager.kinetic_energy_accumulator = ObservableAccumulator{T}()
    manager.potential_energy_accumulator = ObservableAccumulator{T}()
    manager.spin_correlation_accumulator = ObservableAccumulator{T}()
    manager.density_correlation_accumulator = ObservableAccumulator{T}()

    # Reset custom observables
    for (name, acc) in manager.custom_observables
        manager.custom_observables[name] = ObservableAccumulator{T}()
    end

    # Reset statistics
    manager.total_measurements = 0
    manager.measurement_time = 0.0
end

"""
    benchmark_observables(n_site::Int = 10, n_elec::Int = 5, n_iterations::Int = 1000)

Benchmark observable measurements.
"""
function benchmark_observables(n_site::Int = 10, n_elec::Int = 5, n_iterations::Int = 1000)
    println(
        "Benchmarking observable measurements (n_site=$n_site, n_elec=$n_elec, iterations=$n_iterations)...",
    )

    # Create observable manager
    manager = ObservableManager{ComplexF64}(n_site, n_elec)

    # Initialize electron configuration
    ele_idx = collect(1:n_elec)
    ele_cfg = zeros(Int, n_site)
    ele_cfg[1:n_elec] .= 1

    # Benchmark measurements
    @time begin
        for _ = 1:n_iterations
            measure_observables!(manager, ele_idx, ele_cfg, 1.0)
        end
    end
    println("  Observable measurement rate")

    # Print statistics
    energy_stats = get_observable_statistics(manager, "energy")
    println("Observable benchmark completed.")
    println("  Total measurements: $(manager.total_measurements)")
    println("  Energy mean: $(energy_stats.mean)")
    println("  Energy standard error: $(energy_stats.standard_error)")
end
