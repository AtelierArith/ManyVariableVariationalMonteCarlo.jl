@testitem "sampler" begin

    @testset "VMCConfig basic functionality" begin
        using Test
        using ManyVariableVariationalMonteCarlo
        using StableRNGs
        using Random
        # Test VMCConfig creation with defaults
        config = VMCConfig()
        @test config.n_samples == 1000
        @test config.n_thermalization == 100
        @test config.n_measurement == 100
        @test config.n_update_per_sample == 1
        @test config.acceptance_target == 0.5
        @test config.temperature == 1.0
        @test !config.use_two_electron_updates
        @test config.two_electron_probability == 0.1
        # Test VMCConfig creation with custom parameters
        config = VMCConfig(;
            n_samples = 500,
            n_thermalization = 50,
            n_measurement = 50,
            n_update_per_sample = 2,
            acceptance_target = 0.6,
            temperature = 0.5,
            use_two_electron_updates = true,
            two_electron_probability = 0.2,
        )
        @test config.n_samples == 500
        @test config.n_thermalization == 50
        @test config.n_measurement == 50
        @test config.n_update_per_sample == 2
        @test config.acceptance_target == 0.6
        @test config.temperature == 0.5
        @test config.use_two_electron_updates
        @test config.two_electron_probability == 0.2
    end
    @testset "VMCState basic functionality" begin
        using Test
        using ManyVariableVariationalMonteCarlo
        using StableRNGs
        using Random
        # Test VMCState creation
        state = VMCState{ComplexF64}(4, 8)
        @test state.n_electrons == 4
        @test state.n_sites == 8
        @test length(state.electron_positions) == 4
        @test state.wavefunction_value == 0.0
        @test state.log_wavefunction_value == 0.0
        @test state.n_accepted == 0
        @test state.n_rejected == 0
        @test state.n_updates == 0
    end
    @testset "VMCState initialization" begin
        using Test
        using ManyVariableVariationalMonteCarlo
        using StableRNGs
        using Random
        # Test VMCState initialization
        state = VMCState{ComplexF64}(3, 6)
        initial_positions = [1, 3, 5]
        initialize_vmc_state!(state, initial_positions)
        @test state.electron_positions == initial_positions
        @test state.n_accepted == 0
        @test state.n_rejected == 0
        @test state.n_updates == 0
    end
    @testset "VMCState move proposals" begin
        using Test
        using ManyVariableVariationalMonteCarlo
        using StableRNGs
        using Random
        # Test move proposals
        state = VMCState{ComplexF64}(3, 6)
        initial_positions = [1, 3, 5]
        initialize_vmc_state!(state, initial_positions)
        rng = MersenneTwister(12345)
        # Test single electron move proposal
        electron_idx, new_pos, old_pos = propose_single_electron_move(state, rng)
        @test 1 <= electron_idx <= 3
        @test 1 <= new_pos <= 6
        @test old_pos == state.electron_positions[electron_idx]
        @test new_pos != old_pos
        # Test two electron move proposal
        electron1_idx, new_pos1, old_pos1, electron2_idx, new_pos2, old_pos2 =
            propose_two_electron_move(state, rng)
        @test 1 <= electron1_idx <= 3
        @test 1 <= electron2_idx <= 3
        @test electron1_idx != electron2_idx
        @test 1 <= new_pos1 <= 6
        @test 1 <= new_pos2 <= 6
        @test old_pos1 == state.electron_positions[electron1_idx]
        @test old_pos2 == state.electron_positions[electron2_idx]
        @test new_pos1 != old_pos1
        @test new_pos2 != old_pos2
    end
    @testset "VMCState move acceptance" begin
        using Test
        using ManyVariableVariationalMonteCarlo
        using StableRNGs
        using Random
        # Test move acceptance
        state = VMCState{ComplexF64}(3, 6)
        initial_positions = [1, 3, 5]
        initialize_vmc_state!(state, initial_positions)
        # Test single electron move acceptance
        move_data = (1, 2, 1)  # electron_idx, new_pos, old_pos
        accept_move!(state, :single_electron, move_data)
        @test state.electron_positions[1] == 2
        @test state.n_accepted == 1
        @test state.n_updates == 1
        # Test two electron move acceptance
        move_data = (1, 4, 2, 2, 6, 3)  # electron1_idx, new_pos1, old_pos1, electron2_idx, new_pos2, old_pos2
        accept_move!(state, :two_electron, move_data)
        @test state.electron_positions[1] == 4
        @test state.electron_positions[2] == 6
        @test state.n_accepted == 2
        @test state.n_updates == 2
    end
    @testset "VMCState move rejection" begin
        using Test
        using ManyVariableVariationalMonteCarlo
        using StableRNGs
        using Random
        # Test move rejection
        state = VMCState{ComplexF64}(3, 6)
        initial_positions = [1, 3, 5]
        initialize_vmc_state!(state, initial_positions)
        reject_move!(state)
        @test state.electron_positions == initial_positions  # Positions unchanged
        @test state.n_accepted == 0
        @test state.n_rejected == 1
        @test state.n_updates == 1
    end
    @testset "VMCState utility functions" begin
        using Test
        using ManyVariableVariationalMonteCarlo
        using StableRNGs
        using Random
        # Test utility functions
        state = VMCState{ComplexF64}(3, 6)
        initial_positions = [1, 3, 5]
        initialize_vmc_state!(state, initial_positions)
        # Test acceptance rate calculation
        @test get_acceptance_rate(state) == 0.0  # No moves yet
        # Simulate some moves
        state.n_accepted = 3
        state.n_updates = 5
        @test get_acceptance_rate(state) == 0.6
        # Test reset
        reset_vmc_state!(state)
        @test state.n_accepted == 0
        @test state.n_rejected == 0
        @test state.n_updates == 0
    end
    @testset "VMCResults basic functionality" begin
        using Test
        using ManyVariableVariationalMonteCarlo
        using StableRNGs
        using Random
        # Test VMCResults creation
        energy_samples = [1.0 + 0.0im, 2.0 + 0.0im, 3.0 + 0.0im]
        observables = Dict(
            "energy" => energy_samples,
            "magnetization" => [0.1 + 0.0im, 0.2 + 0.0im, 0.3 + 0.0im],
        )
        results = VMCResults{ComplexF64}(
            1.5 + 0.0im,
            0.5 + 0.0im,
            energy_samples,
            observables,
            0.6,
            3,
            10,
            3,
            1.0,
            3,
        )
        @test results.energy_mean == 1.5 + 0.0im
        @test results.energy_std == 0.5 + 0.0im
        @test results.energy_samples == energy_samples
        @test results.observables == observables
        @test results.acceptance_rate == 0.6
        @test results.n_samples == 3
        @test results.n_thermalization == 10
        @test results.n_measurement == 3
        @test results.autocorrelation_time == 1.0
        @test results.effective_samples == 3
    end
    @testset "VMC sampling integration" begin
        using Test
        using ManyVariableVariationalMonteCarlo
        using StableRNGs
        using Random
        # Test basic VMC sampling workflow
        state = VMCState{ComplexF64}(2, 4)
        initial_positions = [1, 3]
        initialize_vmc_state!(state, initial_positions)
        config = VMCConfig(;
            n_samples = 10,
            n_thermalization = 5,
            n_measurement = 5,
            n_update_per_sample = 1,
            use_two_electron_updates = false,
        )
        rng = MersenneTwister(12345)
        # Run VMC sampling
        results = run_vmc_sampling!(state, config, rng)
        @test results.n_samples == 5
        @test results.n_thermalization == 5
        @test results.n_measurement == 5
        @test length(results.energy_samples) == 5
        @test haskey(results.observables, "energy")
        @test 0.0 <= results.acceptance_rate <= 1.0
        @test results.effective_samples > 0
    end
end # @testitem "sampler"
