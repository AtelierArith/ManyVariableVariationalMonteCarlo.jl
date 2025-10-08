"""
Random Number Generation module for ManyVariableVariationalMonteCarlo.jl

Provides high-quality random number generation with reproducible seeds
and parallel stream support, inspired by the SFMT implementation in mVMC.
"""

using Random
using StableRNGs

"""
    VMCRng

High-quality random number generator for VMC simulations.
Uses StableRNG for reproducible results across Julia versions.
"""
struct VMCRng
    rng::StableRNG
    seed::UInt32

    function VMCRng(seed::UInt32 = rand(UInt32))
        if seed == 0
            throw(ArgumentError("RNG seed cannot be zero"))
        end
        new(StableRNG(seed), seed)
    end
end

"""
    ParallelRngManager

Manages multiple independent RNG streams for parallel computation.
Each thread/process gets its own RNG to avoid synchronization overhead.
"""
struct ParallelRngManager
    rngs::Vector{VMCRng}
    master_seed::UInt32

    function ParallelRngManager(master_seed::UInt32, n_streams::Int = Threads.nthreads())
        # Generate seeds for each stream using deterministic sequence
        seeds = generate_stream_seeds(master_seed, n_streams)
        rngs = [VMCRng(seed) for seed in seeds]
        new(rngs, master_seed)
    end
end

"""
    generate_stream_seeds(master_seed::UInt32, n_streams::Int)

Generate deterministic sequence of seeds for parallel streams.
Ensures different streams are statistically independent.

C実装参考: sfmt.c 1行目から580行目まで
"""
function generate_stream_seeds(master_seed::UInt32, n_streams::Int)
    master_rng = StableRNG(master_seed)
    return [rand(master_rng, UInt32) for _ = 1:n_streams]
end

# Global RNG manager
const GLOBAL_RNG_MANAGER = Ref{Union{ParallelRngManager,Nothing}}(nothing)

"""
    initialize_rng!(seed::UInt32 = rand(UInt32), n_streams::Int = Threads.nthreads())

Initialize global RNG manager with given seed and number of streams.
Call this once at the beginning of simulation.
"""
function initialize_rng!(seed::UInt32 = rand(UInt32), n_streams::Int = Threads.nthreads())
    GLOBAL_RNG_MANAGER[] = ParallelRngManager(seed, n_streams)
    return seed
end

"""
    get_thread_rng(thread_id::Int = Threads.threadid())

Get RNG for specific thread. Thread-safe access to independent streams.
"""
function get_thread_rng(thread_id::Int = Threads.threadid())
    if thread_id < 1
        throw(ArgumentError("Thread ID must be positive, got $thread_id"))
    end

    manager = GLOBAL_RNG_MANAGER[]
    if manager === nothing
        error("RNG not initialized. Call initialize_rng!() first.")
    end

    if thread_id > length(manager.rngs)
        error("Thread ID $thread_id exceeds number of RNG streams $(length(manager.rngs))")
    end

    return manager.rngs[thread_id].rng
end

"""
    vmcrand(thread_id::Int = Threads.threadid())

Generate random Float64 ∈ [0,1) for current thread.
High-performance interface for Monte Carlo sampling.
"""
function vmcrand(thread_id::Int = Threads.threadid())
    rng = get_thread_rng(thread_id)
    return rand(rng)
end

"""
    vmcrandn(thread_id::Int = Threads.threadid())

Generate random normal variate for current thread.
"""
function vmcrandn(thread_id::Int = Threads.threadid())
    rng = get_thread_rng(thread_id)
    return randn(rng)
end

"""
    vmcrand_int(max_val::Int, thread_id::Int = Threads.threadid())

Generate random integer in range [0, max_val).
Optimized for uniform sampling of discrete choices.
"""
function vmcrand_int(max_val::Int, thread_id::Int = Threads.threadid())
    rng = get_thread_rng(thread_id)
    return rand(rng, 0:(max_val-1))
end

"""
    vmcrand_bool(prob::Float64 = 0.5, thread_id::Int = Threads.threadid())

Generate random boolean with given probability.
Optimized for Metropolis acceptance/rejection.
"""
function vmcrand_bool(prob::Float64 = 0.5, thread_id::Int = Threads.threadid())
    return vmcrand(thread_id) < prob
end

"""
    RngState

Captures the state of all RNG streams for checkpointing.
"""
struct RngState
    master_seed::UInt32
    stream_seeds::Vector{UInt32}

    function RngState(manager::ParallelRngManager)
        seeds = [rng.seed for rng in manager.rngs]
        new(manager.master_seed, seeds)
    end
end

"""
    save_rng_state()

Save current RNG state for checkpointing/restart.
"""
function save_rng_state()
    manager = GLOBAL_RNG_MANAGER[]
    if manager === nothing
        error("RNG not initialized.")
    end
    return RngState(manager)
end

"""
    restore_rng_state!(state::RngState)

Restore RNG state from checkpoint.
"""
function restore_rng_state!(state::RngState)
    if !isa(state, RngState)
        throw(
            ArgumentError(
                "Invalid RNG state type: expected RngState, got $(typeof(state))",
            ),
        )
    end

    # Recreate manager with same configuration
    manager = ParallelRngManager(state.master_seed, length(state.stream_seeds))

    # Restore each stream's seed
    for (i, seed) in enumerate(state.stream_seeds)
        manager.rngs[i] = VMCRng(seed)
    end

    GLOBAL_RNG_MANAGER[] = manager
end

"""
    rng_info()

Print information about current RNG configuration.
"""
function rng_info()
    manager = GLOBAL_RNG_MANAGER[]
    if manager === nothing
        println("RNG not initialized")
        return
    end

    println("RNG Configuration:")
    println("  Master seed: $(manager.master_seed)")
    println("  Number of streams: $(length(manager.rngs))")
    println("  RNG type: StableRNG")

    for (i, rng) in enumerate(manager.rngs)
        println("  Stream $i seed: $(rng.seed)")
    end
end

"""
    benchmark_rng(n_samples::Int = 10^6)

Benchmark RNG performance for VMC-typical usage patterns.
"""
function benchmark_rng(n_samples::Int = 10^6)
    if GLOBAL_RNG_MANAGER[] === nothing
        initialize_rng!()
    end

    println("Benchmarking RNG performance ($n_samples samples)...")

    # Benchmark uniform random numbers
    @time begin
        for _ = 1:n_samples
            vmcrand()
        end
    end
    println("  Uniform random generation rate")

    # Benchmark random integers
    @time begin
        for _ = 1:n_samples
            vmcrand_int(100)
        end
    end
    println("  Random integer generation rate")

    # Benchmark boolean decisions
    @time begin
        for _ = 1:n_samples
            vmcrand_bool(0.5)
        end
    end
    println("  Boolean decision generation rate")

    # Benchmark normal variates
    @time begin
        for _ = 1:n_samples
            vmcrandn()
        end
    end
    println("  Normal variate generation rate")
end

"""
    test_rng_quality(n_samples::Int = 10^5)

Basic statistical tests for RNG quality.
"""
function test_rng_quality(n_samples::Int = 10^5)
    if GLOBAL_RNG_MANAGER[] === nothing
        initialize_rng!(12345)  # Fixed seed for reproducible tests
    end

    println("Testing RNG quality ($n_samples samples)...")

    # Test uniform distribution
    samples = [vmcrand() for _ = 1:n_samples]
    mean_val = sum(samples) / n_samples
    println("  Uniform mean: $mean_val (expected: 0.5)")

    # Test normal distribution
    normal_samples = [vmcrandn() for _ = 1:n_samples]
    normal_mean = sum(normal_samples) / n_samples
    normal_var = sum(x^2 for x in normal_samples) / n_samples - normal_mean^2
    println("  Normal mean: $normal_mean (expected: 0.0)")
    println("  Normal variance: $normal_var (expected: 1.0)")

    # Test integer distribution
    int_samples = [vmcrand_int(10) for _ = 1:n_samples]
    int_counts = zeros(Int, 10)
    for sample in int_samples
        int_counts[sample+1] += 1
    end
    expected_count = n_samples / 10
    max_deviation = maximum(abs(count - expected_count) for count in int_counts)
    println("  Integer uniformity max deviation: $max_deviation (lower is better)")

    println("RNG quality test completed.")
end
