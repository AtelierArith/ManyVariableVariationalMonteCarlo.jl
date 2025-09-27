@testitem "VMCRng basic functionality" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    # Test RNG creation with specific seed
    rng = VMCRng(UInt32(12345))
    @test rng.seed == 12345
    # Test random seed creation
    rng2 = VMCRng()
    @test isa(rng2.seed, UInt32)
end
@testitem "ParallelRngManager" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    using ManyVariableVariationalMonteCarlo: ParallelRngManager, generate_stream_seeds
    # Test seed generation
    seeds = generate_stream_seeds(UInt32(12345), 4)
    @test length(seeds) == 4
    @test all(isa(s, UInt32) for s in seeds)
    @test length(unique(seeds)) == 4  # All seeds should be different
    # Test manager creation
    manager = ParallelRngManager(UInt32(12345), 4)
    @test length(manager.rngs) == 4
    @test manager.master_seed == 12345
end
@testitem "RNG initialization and access" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    # Test initialization
    seed = initialize_rng!(UInt32(54321), 2)
    @test seed == 54321
    # Test thread RNG access
    rng1 = get_thread_rng(1)
    rng2 = get_thread_rng(2)
    @test rng1 !== rng2  # Should be different objects
    # Test error on invalid thread ID
    @test_throws Exception get_thread_rng(5)
end
@testitem "VMC random number functions" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    initialize_rng!(UInt32(11111), 1)
    # Test basic random generation
    r1 = vmcrand(1)
    r2 = vmcrand(1)
    @test 0.0 <= r1 < 1.0
    @test 0.0 <= r2 < 1.0
    @test r1 != r2
    # Test normal random generation
    n1 = vmcrandn(1)
    n2 = vmcrandn(1)
    @test isa(n1, Float64)
    @test isa(n2, Float64)
    @test n1 != n2
    # Test integer random generation
    i1 = vmcrand_int(10, 1)
    i2 = vmcrand_int(10, 1)
    @test 0 <= i1 < 10
    @test 0 <= i2 < 10
    @test isa(i1, Int)
    # Test boolean random generation
    b1 = vmcrand_bool(0.5, 1)
    b2 = vmcrand_bool(1.0, 1)
    b3 = vmcrand_bool(0.0, 1)
    @test isa(b1, Bool)
    @test b2 == true
    @test b3 == false
end
@testitem "RNG reproducibility" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    # Test that same seed produces same sequence
    initialize_rng!(UInt32(99999), 1)
    seq1 = [vmcrand(1) for _ in 1:10]
    initialize_rng!(UInt32(99999), 1)
    seq2 = [vmcrand(1) for _ in 1:10]
    @test seq1 == seq2
end
@testitem "RNG state save/restore" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    # Initialize RNG system
    initialize_rng!(UInt32(12345), 1)
    # Test that save/restore functions work without error
    state = save_rng_state()
    @test isa(state, RngState)
    # Test restore works
    restore_rng_state!(state)
    @test true  # If we get here, no error was thrown
end
@testitem "RNG error handling" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    # Test invalid RNG initialization
    @test_throws ArgumentError VMCRng(UInt32(0))
    # Test invalid thread ID
    @test_throws ArgumentError get_thread_rng(-1)
    # Test RNG with invalid state
    invalid_state = "invalid_state"
    @test_throws MethodError restore_rng_state!(invalid_state)
end
@testitem "RNG statistical properties" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    # Test uniform distribution properties
    initialize_rng!(UInt32(33333), 1)
    # Generate large sample
    n_samples = 10000
    samples = [vmcrand(1) for _ in 1:n_samples]
    # Test range
    @test all(0.0 <= x < 1.0 for x in samples)
    # Test approximate mean (should be around 0.5)
    mean_val = sum(samples) / n_samples
    @test 0.45 <= mean_val <= 0.55
    # Test integer distribution
    int_samples = [vmcrand_int(10, 1) for _ in 1:n_samples]
    @test all(0 <= x < 10 for x in int_samples)
    # Count occurrences of each integer
    counts = zeros(Int, 10)
    for x in int_samples
        counts[x+1] += 1
    end
    # Each should appear roughly n_samples/10 times
    expected = n_samples / 10
    @test all(0.8 * expected <= count <= 1.2 * expected for count in counts)
end
@testitem "RNG thread safety" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    initialize_rng!(UInt32(44444), 4)
    # Test that different threads get different sequences
    results = Vector{Vector{Float64}}(undef, 4)
    # Simulate parallel access
    for tid in 1:4
        results[tid] = [vmcrand(tid) for _ in 1:100]
    end
    # All sequences should be different
    for i in 1:4
        for j in (i+1):4
            @test results[i] != results[j]
        end
    end
end
@testitem "RNG performance characteristics" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    initialize_rng!(UInt32(55555), 1)
    # Test that random generation is fast
    n_samples = 10000
    # Time uniform generation
    time_uniform = @elapsed for _ in 1:n_samples
        vmcrand(1)
    end
    @test time_uniform < 0.1  # Should be very fast
    # Time integer generation
    time_int = @elapsed for _ in 1:n_samples
        vmcrand_int(100, 1)
    end
    @test time_int < 0.1
    # Time boolean generation
    time_bool = @elapsed for _ in 1:n_samples
        vmcrand_bool(0.5, 1)
    end
    @test time_bool < 0.1
end