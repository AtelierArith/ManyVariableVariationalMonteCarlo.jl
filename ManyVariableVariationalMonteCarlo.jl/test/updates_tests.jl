@testitem "updates" begin
    """
    Tests for Monte Carlo update algorithms

    Tests all update algorithm functionality including:
    - Single electron updates
    - Two electron updates
    - Exchange hopping updates
    - Update manager
    - Performance benchmarks
    """

    using Test
    using Random
    using StableRNGs
    using ManyVariableVariationalMonteCarlo

    @testset "Single Electron Update Basic Functionality" begin

        # Test single electron update creation
        update = SingleElectronUpdate{ComplexF64}(4, 2)
        @test update.n_site == 4
        @test update.n_elec == 2
        @test update.max_hop_distance == 1
        @test update.update_probability == 0.1
        @test update.total_attempts == 0
        @test update.total_accepted == 0
        @test update.acceptance_rate == 0.0

        # Test electron configuration
        ele_idx = [1, 2]
        ele_cfg = [1, 1, 0, 0]

        # Test move proposal
        rng = StableRNG(123)
        result = propose_single_electron_move(update, ele_idx, ele_cfg, rng)
        @test isa(result, UpdateResult{ComplexF64})
        @test result.update_type == SINGLE_ELECTRON
        @test length(result.electron_indices) == 1
        @test length(result.site_indices) == 2
    end

    @testset "Two Electron Update Basic Functionality" begin

        # Test two electron update creation
        update = TwoElectronUpdate{ComplexF64}(4, 2)
        @test update.n_site == 4
        @test update.n_elec == 2
        @test update.max_hop_distance == 1
        @test update.update_probability == 0.05
        @test update.total_attempts == 0
        @test update.total_accepted == 0
        @test update.acceptance_rate == 0.0

        # Test electron configuration
        ele_idx = [1, 2]
        ele_cfg = [1, 1, 0, 0]

        # Test move proposal
        rng = StableRNG(123)
        result = propose_two_electron_move(update, ele_idx, ele_cfg, rng)
        @test isa(result, UpdateResult{ComplexF64})
        @test result.update_type == TWO_ELECTRON
        @test length(result.electron_indices) == 2
        @test length(result.site_indices) == 4
    end

    @testset "Exchange Hopping Update Basic Functionality" begin

        # Test exchange hopping update creation
        update = ExchangeHoppingUpdate{ComplexF64}(4, 2)
        @test update.n_site == 4
        @test update.n_elec == 2
        @test update.max_hop_distance == 1
        @test update.update_probability == 0.02
        @test update.total_attempts == 0
        @test update.total_accepted == 0
        @test update.acceptance_rate == 0.0

        # Test electron configuration
        ele_idx = [1, 2]
        ele_cfg = [1, 1, 0, 0]

        # Test move proposal
        rng = StableRNG(123)
        result = propose_exchange_hopping(update, ele_idx, ele_cfg, rng)
        @test isa(result, UpdateResult{ComplexF64})
        @test result.update_type == EXCHANGE_HOPPING
        @test length(result.electron_indices) == 2
        @test length(result.site_indices) == 4
    end

    @testset "Update Manager Basic Functionality" begin

        # Test update manager creation
        manager = UpdateManager{ComplexF64}(4, 2)
        @test manager.single_electron.n_site == 4
        @test manager.two_electron.n_site == 4
        @test manager.exchange_hopping.n_site == 4
        @test manager.single_electron_prob == 0.7
        @test manager.two_electron_prob == 0.2
        @test manager.exchange_hopping_prob == 0.1
        @test manager.total_attempts == 0
        @test manager.total_accepted == 0
        @test manager.overall_acceptance_rate == 0.0

        # Test electron configuration
        ele_idx = [1, 2]
        ele_cfg = [1, 1, 0, 0]

        # Test move proposal
        rng = StableRNG(123)
        result = propose_update(manager, ele_idx, ele_cfg, rng)
        @test isa(result, UpdateResult{ComplexF64})
        @test result.update_type in [SINGLE_ELECTRON, TWO_ELECTRON, EXCHANGE_HOPPING]
    end

    @testset "Update Manager Statistics" begin

        # Test update manager statistics
        manager = UpdateManager{ComplexF64}(4, 2)

        # Create a successful result
        result = UpdateResult{ComplexF64}(true, ComplexF64(0.5), SINGLE_ELECTRON)
        result.electron_indices = [1]
        result.site_indices = [1, 2]

        # Test acceptance
        accept_update!(manager, result)
        @test manager.total_attempts == 1
        @test manager.total_accepted == 1
        @test manager.overall_acceptance_rate == 1.0
        @test manager.single_electron.total_attempts == 1
        @test manager.single_electron.total_accepted == 1
        @test manager.single_electron.acceptance_rate == 1.0

        # Test rejection
        reject_update!(manager, result)
        @test manager.total_attempts == 2
        @test manager.total_accepted == 1
        @test manager.overall_acceptance_rate == 0.5
        @test manager.single_electron.total_attempts == 2
        @test manager.single_electron.total_accepted == 1
        @test manager.single_electron.acceptance_rate == 0.5
    end

    @testset "Update Manager Statistics Retrieval" begin

        # Test statistics retrieval
        manager = UpdateManager{ComplexF64}(4, 2)

        # Add some statistics
        result = UpdateResult{ComplexF64}(true, ComplexF64(0.5), SINGLE_ELECTRON)
        result.electron_indices = [1]
        result.site_indices = [1, 2]

        accept_update!(manager, result)
        reject_update!(manager, result)

        # Test statistics retrieval
        stats = get_update_statistics(manager)
        @test stats.overall.total_attempts == 2
        @test stats.overall.total_accepted == 1
        @test stats.overall.acceptance_rate == 0.5
        @test stats.single_electron.total_attempts == 2
        @test stats.single_electron.total_accepted == 1
        @test stats.single_electron.acceptance_rate == 0.5
    end

    @testset "Update Manager Statistics Reset" begin

        # Test statistics reset
        manager = UpdateManager{ComplexF64}(4, 2)

        # Add some statistics
        result = UpdateResult{ComplexF64}(true, ComplexF64(0.5), SINGLE_ELECTRON)
        result.electron_indices = [1]
        result.site_indices = [1, 2]

        accept_update!(manager, result)
        reject_update!(manager, result)

        # Reset statistics
        reset_update_statistics!(manager)
        @test manager.total_attempts == 0
        @test manager.total_accepted == 0
        @test manager.overall_acceptance_rate == 0.0
        @test manager.single_electron.total_attempts == 0
        @test manager.single_electron.total_accepted == 0
        @test manager.single_electron.acceptance_rate == 0.0
    end

    @testset "Update Performance" begin

        # Test performance with larger systems
        manager = UpdateManager{ComplexF64}(10, 5)

        ele_idx = collect(1:5)
        ele_cfg = zeros(Int, 10)
        ele_cfg[1:5] .= 1

        # Benchmark update proposals
        n_iterations = 1000
        @time begin
            for _ = 1:n_iterations
                result = propose_update(manager, ele_idx, ele_cfg, Random.GLOBAL_RNG)
                if result.success
                    # Simulate acceptance/rejection
                    if rand() < result.acceptance_probability
                        accept_update!(manager, result)
                    else
                        reject_update!(manager, result)
                    end
                end
            end
        end

        @test manager.total_attempts > 0
    end

    @testset "Update Edge Cases" begin

        # Test edge cases
        update = SingleElectronUpdate{ComplexF64}(4, 0)  # No electrons
        ele_idx = Int[]
        ele_cfg = [0, 0, 0, 0]

        rng = StableRNG(123)
        result = propose_single_electron_move(update, ele_idx, ele_cfg, rng)
        @test !result.success
        @test result.ratio == zero(ComplexF64)

        # Test with no available sites
        update = SingleElectronUpdate{ComplexF64}(2, 2)
        ele_idx = [1, 2]
        ele_cfg = [1, 1]  # All sites occupied

        result = propose_single_electron_move(update, ele_idx, ele_cfg, rng)
        @test !result.success
    end

    @testset "Update Complex vs Real" begin

        # Test complex updates
        update_complex = SingleElectronUpdate{ComplexF64}(4, 2)
        ele_idx = [1, 2]
        ele_cfg = [1, 1, 0, 0]

        rng = StableRNG(123)
        result_complex = propose_single_electron_move(update_complex, ele_idx, ele_cfg, rng)
        @test isa(result_complex.ratio, ComplexF64)

        # Test real updates
        update_real = SingleElectronUpdate{Float64}(4, 2)
        result_real = propose_single_electron_move(update_real, ele_idx, ele_cfg, rng)
        @test isa(result_real.ratio, Float64)
    end

    @testset "Update Manager Probabilities" begin

        # Test update manager with different probabilities
        manager = UpdateManager{ComplexF64}(4, 2)
        manager.single_electron_prob = 0.5
        manager.two_electron_prob = 0.3
        manager.exchange_hopping_prob = 0.2

        ele_idx = [1, 2]
        ele_cfg = [1, 1, 0, 0]

        # Test multiple proposals to check probability distribution
        n_proposals = 1000
        single_count = 0
        two_count = 0
        exchange_count = 0

        for _ = 1:n_proposals
            result = propose_update(manager, ele_idx, ele_cfg, Random.GLOBAL_RNG)
            if result.success
                if result.update_type == SINGLE_ELECTRON
                    single_count += 1
                elseif result.update_type == TWO_ELECTRON
                    two_count += 1
                elseif result.update_type == EXCHANGE_HOPPING
                    exchange_count += 1
                end
            end
        end

        # Check that probabilities are approximately correct
        @test single_count > 0
        @test two_count > 0
        # Exchange hopping should occur occasionally with PBC-aware updates
        @test exchange_count > 0
    end
end # @testitem "updates"
