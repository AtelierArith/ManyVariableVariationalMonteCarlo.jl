@testitem "observables" begin
    """
    Tests for observable measurement system

    Tests all observable measurement functionality including:
    - Energy calculations
    - Correlation functions
    - Observable accumulators
    - Observable manager
    - Performance benchmarks
    """

    using Test
    using StableRNGs

    @testset "Observable Measurement Basic Functionality" begin

        # Test observable measurement creation
        measurement = ObservableMeasurement{ComplexF64}(ENERGY, ComplexF64(1.5), 1.0)
        @test measurement.observable_type == ENERGY
        @test measurement.value == ComplexF64(1.5)
        @test measurement.weight == 1.0
        @test measurement.measurement_time == 0.0
    end

    @testset "Observable Accumulator Basic Functionality" begin

        # Test observable accumulator creation
        accumulator = ObservableAccumulator{ComplexF64}()
        @test accumulator.sum_values == zero(ComplexF64)
        @test accumulator.sum_squared_values == zero(ComplexF64)
        @test accumulator.sum_weights == 0.0
        @test accumulator.sum_weighted_values == zero(ComplexF64)
        @test accumulator.sum_weighted_squared_values == zero(ComplexF64)
        @test accumulator.mean_value == zero(ComplexF64)
        @test accumulator.variance == zero(ComplexF64)
        @test accumulator.standard_error == zero(ComplexF64)
        @test accumulator.n_measurements == 0
        @test isempty(accumulator.measurements)
    end

    @testset "Observable Accumulator Measurements" begin

        # Test adding measurements
        accumulator = ObservableAccumulator{ComplexF64}()

        # Add first measurement
        measurement1 = ObservableMeasurement{ComplexF64}(ENERGY, ComplexF64(1.0), 1.0)
        add_measurement!(accumulator, measurement1)
        @test accumulator.n_measurements == 1
        @test accumulator.sum_values == ComplexF64(1.0)
        @test accumulator.sum_weights == 1.0
        @test accumulator.mean_value == ComplexF64(1.0)

        # Add second measurement
        measurement2 = ObservableMeasurement{ComplexF64}(ENERGY, ComplexF64(2.0), 1.0)
        add_measurement!(accumulator, measurement2)
        @test accumulator.n_measurements == 2
        @test accumulator.sum_values == ComplexF64(3.0)
        @test accumulator.sum_weights == 2.0
        @test accumulator.mean_value == ComplexF64(1.5)
    end

    @testset "Energy Calculator Basic Functionality" begin

        # Test energy calculator creation
        calc = EnergyCalculator{ComplexF64}(4, 2)
        @test calc.n_site == 4
        @test calc.n_elec == 2
        @test size(calc.hopping_matrix) == (4, 4)
        @test size(calc.interaction_matrix) == (4, 4)
        @test length(calc.external_potential) == 4
        @test calc.total_calculations == 0
        @test calc.calculation_time == 0.0

        # Test electron configuration
        ele_idx = [1, 2]
        ele_cfg = [1, 1, 0, 0]

        # Test energy calculations
        kinetic = calculate_kinetic_energy(calc, ele_idx, ele_cfg)
        @test isa(kinetic, ComplexF64)
        @test calc.total_calculations == 1

        potential = calculate_potential_energy(calc, ele_idx, ele_cfg)
        @test isa(potential, ComplexF64)
        @test calc.total_calculations == 2

        total = calculate_total_energy(calc, ele_idx, ele_cfg)
        @test isa(total, ComplexF64)
        @test calc.total_calculations == 4  # 1 kinetic + 1 potential + 2 more inside total_energy
        @test total == kinetic + potential
    end

    @testset "Correlation Function Basic Functionality" begin

        # Test correlation function creation
        corr = CorrelationFunction{ComplexF64}(4, 2)
        @test corr.n_site == 4
        @test corr.n_elec == 2
        @test corr.max_distance == 5
        @test corr.correlation_type == "spin"
        @test length(corr.correlation_buffer) == 6
        @test corr.total_calculations == 0
        @test corr.calculation_time == 0.0

        # Test electron configuration
        ele_idx = [1, 2]
        ele_cfg = [1, 1, 0, 0]

        # Test spin correlation calculation
        spin_corr = calculate_spin_correlation(corr, ele_idx, ele_cfg)
        @test isa(spin_corr, Vector{ComplexF64})
        @test length(spin_corr) == 6
        @test corr.total_calculations == 1

        # Test density correlation calculation
        density_corr = calculate_density_correlation(corr, ele_idx, ele_cfg)
        @test isa(density_corr, Vector{ComplexF64})
        @test length(density_corr) == 6
        @test corr.total_calculations == 2
    end

    @testset "Observable Manager Basic Functionality" begin

        # Test observable manager creation
        manager = ObservableManager{ComplexF64}(4, 2)
        @test manager.energy_calculator.n_site == 4
        @test manager.energy_calculator.n_elec == 2
        @test manager.spin_correlation.n_site == 4
        @test manager.spin_correlation.n_elec == 2
        @test manager.density_correlation.n_site == 4
        @test manager.density_correlation.n_elec == 2
        @test isempty(manager.custom_observables)
        @test manager.total_measurements == 0
        @test manager.measurement_time == 0.0

        # Test electron configuration
        ele_idx = [1, 2]
        ele_cfg = [1, 1, 0, 0]

        # Test observable measurement
        measure_observables!(manager, ele_idx, ele_cfg, 1.0)
        @test manager.total_measurements == 1
        @test manager.energy_accumulator.n_measurements > 0
        @test manager.kinetic_energy_accumulator.n_measurements > 0
        @test manager.potential_energy_accumulator.n_measurements > 0
    end

    @testset "Custom Observables" begin

        # Test custom observables
        manager = ObservableManager{ComplexF64}(4, 2)

        # Add custom observable
        add_custom_observable!(manager, "custom_obs")
        @test haskey(manager.custom_observables, "custom_obs")

        # Measure custom observable
        measure_custom_observable!(manager, "custom_obs", ComplexF64(2.5), 1.0)
        @test manager.custom_observables["custom_obs"].n_measurements == 1
        @test manager.custom_observables["custom_obs"].mean_value == ComplexF64(2.5)

        # Test non-existent custom observable
        @test_throws ArgumentError measure_custom_observable!(
            manager,
            "nonexistent",
            ComplexF64(1.0),
        )
    end

    @testset "Observable Statistics" begin

        # Test observable statistics
        manager = ObservableManager{ComplexF64}(4, 2)

        # Add some measurements
        ele_idx = [1, 2]
        ele_cfg = [1, 1, 0, 0]

        for _ = 1:10
            measure_observables!(manager, ele_idx, ele_cfg, 1.0)
        end

        # Test statistics retrieval
        energy_stats = get_observable_statistics(manager, "energy")
        @test energy_stats.n_measurements > 0
        @test isa(energy_stats.mean, ComplexF64)
        @test isa(energy_stats.variance, ComplexF64)
        @test isa(energy_stats.standard_error, ComplexF64)

        kinetic_stats = get_observable_statistics(manager, "kinetic_energy")
        @test kinetic_stats.n_measurements > 0

        potential_stats = get_observable_statistics(manager, "potential_energy")
        @test potential_stats.n_measurements > 0
    end

    @testset "Observable Statistics Reset" begin

        # Test statistics reset
        manager = ObservableManager{ComplexF64}(4, 2)

        # Add some measurements
        ele_idx = [1, 2]
        ele_cfg = [1, 1, 0, 0]

        for _ = 1:5
            measure_observables!(manager, ele_idx, ele_cfg, 1.0)
        end

        # Reset statistics
        reset_observable_statistics!(manager)
        @test manager.total_measurements == 0
        @test manager.measurement_time == 0.0
        @test manager.energy_accumulator.n_measurements == 0
        @test manager.kinetic_energy_accumulator.n_measurements == 0
        @test manager.potential_energy_accumulator.n_measurements == 0
    end

    @testset "Observable Performance" begin

        # Test performance with larger systems
        manager = ObservableManager{ComplexF64}(10, 5)

        ele_idx = collect(1:5)
        ele_cfg = zeros(Int, 10)
        ele_cfg[1:5] .= 1

        # Benchmark measurements
        n_iterations = 1000
        @time begin
            for _ = 1:n_iterations
                measure_observables!(manager, ele_idx, ele_cfg, 1.0)
            end
        end

        @test manager.total_measurements == n_iterations
    end

    @testset "Observable Edge Cases" begin

        # Test edge cases
        manager = ObservableManager{ComplexF64}(4, 0)  # No electrons
        ele_idx = Int[]
        ele_cfg = [0, 0, 0, 0]

        measure_observables!(manager, ele_idx, ele_cfg, 1.0)
        @test manager.total_measurements == 1

        # Test statistics for non-existent observable
        @test_throws ArgumentError get_observable_statistics(manager, "nonexistent")
    end

    @testset "Observable Complex vs Real" begin

        # Test complex observables
        manager_complex = ObservableManager{ComplexF64}(4, 2)
        ele_idx = [1, 2]
        ele_cfg = [1, 1, 0, 0]

        measure_observables!(manager_complex, ele_idx, ele_cfg, 1.0)
        energy_stats_complex = get_observable_statistics(manager_complex, "energy")
        @test isa(energy_stats_complex.mean, ComplexF64)

        # Test real observables
        manager_real = ObservableManager{Float64}(4, 2)
        measure_observables!(manager_real, ele_idx, ele_cfg, 1.0)
        energy_stats_real = get_observable_statistics(manager_real, "energy")
        @test isa(energy_stats_real.mean, Float64)
    end

    @testset "Observable Accumulator Statistics" begin

        # Test accumulator statistics calculation
        accumulator = ObservableAccumulator{ComplexF64}()

        # Add measurements with different weights
        measurement1 = ObservableMeasurement{ComplexF64}(ENERGY, ComplexF64(1.0), 0.5)
        measurement2 = ObservableMeasurement{ComplexF64}(ENERGY, ComplexF64(2.0), 1.5)
        measurement3 = ObservableMeasurement{ComplexF64}(ENERGY, ComplexF64(3.0), 1.0)

        add_measurement!(accumulator, measurement1)
        add_measurement!(accumulator, measurement2)
        add_measurement!(accumulator, measurement3)

        @test accumulator.n_measurements == 3
        @test accumulator.sum_weights == 3.0
        @test accumulator.sum_weighted_values == ComplexF64(6.5)  # 0.5*1 + 1.5*2 + 1.0*3
        @test accumulator.mean_value == ComplexF64(6.5 / 3.0)
    end
end # @testitem "observables"
