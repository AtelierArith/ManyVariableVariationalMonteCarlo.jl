"""
Parallel Monte Carlo Sampling for ManyVariableVariationalMonteCarlo.jl

Implements parallel and distributed Monte Carlo sampling including:
- Multi-threaded sampling
- Distributed computing across multiple processes
- Load balancing and work distribution
- Result aggregation and synchronization
- Performance monitoring and optimization

Ported from parallel sampling concepts in the C reference implementation.
"""

using Distributed
using SharedArrays
using LinearAlgebra
using Random
using StableRNGs
using Statistics

"""
    ParallelConfig

Configuration for parallel Monte Carlo sampling.
"""
mutable struct ParallelConfig
    # Parallel execution settings
    use_threading::Bool
    use_distributed::Bool
    n_threads::Int
    n_processes::Int

    # Work distribution
    work_chunk_size::Int
    load_balancing::Bool
    dynamic_scheduling::Bool

    # Communication settings
    communication_interval::Int
    synchronization_method::String

    # Performance monitoring
    enable_profiling::Bool
    collect_timing_stats::Bool

    function ParallelConfig(;
        use_threading::Bool = true,
        use_distributed::Bool = false,
        n_threads::Int = Threads.nthreads(),
        n_processes::Int = 1,
        work_chunk_size::Int = 1000,
        load_balancing::Bool = true,
        dynamic_scheduling::Bool = true,
        communication_interval::Int = 100,
        synchronization_method::String = "reduce",
        enable_profiling::Bool = false,
        collect_timing_stats::Bool = true,
    )
        new(
            use_threading,
            use_distributed,
            n_threads,
            n_processes,
            work_chunk_size,
            load_balancing,
            dynamic_scheduling,
            communication_interval,
            synchronization_method,
            enable_profiling,
            collect_timing_stats,
        )
    end
end

"""
    ParallelVMCState{T}

Parallel VMC state with distributed components.
"""
mutable struct ParallelVMCState{T<:Union{Float64,ComplexF64}}
    # Local state
    local_state::Any  # Will be EnhancedVMCState{T}

    # Parallel configuration
    config::ParallelConfig

    # Work distribution
    work_queue::Vector{Int}
    current_work_item::Int
    work_completed::Int

    # Communication buffers
    energy_buffer::SharedArray{T}
    gradient_buffer::SharedArray{T}
    observable_buffer::SharedArray{T}

    # Synchronization
    barrier::Union{Nothing,Any}
    mutex::Union{Nothing,Any}

    # Statistics
    local_acceptance_rate::Float64
    local_energy_mean::T
    local_energy_variance::T

    # Performance monitoring
    sampling_time::Float64
    communication_time::Float64
    synchronization_time::Float64

    function ParallelVMCState{T}(local_state, config::ParallelConfig) where {T}
        n_params = 100  # Default parameter count
        n_observables = 50  # Default observable count

        energy_buffer = SharedArray{T}(config.n_processes)
        gradient_buffer = SharedArray{T}(n_params * config.n_processes)
        observable_buffer = SharedArray{T}(n_observables * config.n_processes)

        new{T}(
            local_state,
            config,
            Int[],
            0,
            0,
            energy_buffer,
            gradient_buffer,
            observable_buffer,
            nothing,
            nothing,
            0.0,
            zero(T),
            zero(T),
            0.0,
            0.0,
            0.0,
        )
    end
end

"""
    ThreadedSampler{T}

Multi-threaded Monte Carlo sampler.
"""
mutable struct ThreadedSampler{T<:Union{Float64,ComplexF64}}
    # Threading configuration
    n_threads::Int
    thread_states::Vector{Any}  # Will be Vector{EnhancedVMCState{T}}

    # Work distribution
    work_queue::Vector{Int}
    work_mutex::Union{Nothing,Any}

    # Results aggregation
    energy_results::Vector{T}
    gradient_results::Matrix{T}
    observable_results::Matrix{T}

    # Statistics
    total_samples::Int
    total_accepted::Int
    thread_times::Vector{Float64}

    function ThreadedSampler{T}(
        n_threads::Int,
        n_samples::Int,
        n_params::Int,
        n_observables::Int,
    ) where {T}
        thread_states = Any[]
        work_queue = collect(1:n_samples)
        energy_results = Vector{T}(undef, n_samples)
        gradient_results = Matrix{T}(undef, n_params, n_samples)
        observable_results = Matrix{T}(undef, n_observables, n_samples)
        thread_times = zeros(Float64, n_threads)

        new{T}(
            n_threads,
            thread_states,
            work_queue,
            nothing,
            energy_results,
            gradient_results,
            observable_results,
            0,
            0,
            thread_times,
        )
    end
end

"""
    DistributedSampler{T}

Distributed Monte Carlo sampler across multiple processes.
"""
mutable struct DistributedSampler{T<:Union{Float64,ComplexF64}}
    # Process configuration
    n_processes::Int
    process_id::Int
    process_states::Vector{Any}  # Will be Vector{EnhancedVMCState{T}}

    # Communication
    energy_channel::Union{Nothing,Any}
    gradient_channel::Union{Nothing,Any}
    observable_channel::Union{Nothing,Any}

    # Work distribution
    work_assignments::Vector{Vector{Int}}
    work_completed::Vector{Int}

    # Results
    global_energy_mean::T
    global_energy_variance::T
    global_gradients::Vector{T}
    global_observables::Vector{T}

    # Statistics
    total_samples::Int
    communication_overhead::Float64

    function DistributedSampler{T}(
        n_processes::Int,
        process_id::Int,
        n_samples::Int,
        n_params::Int,
        n_observables::Int,
    ) where {T}
        process_states = Any[]
        work_assignments = [Vector{Int}() for _ = 1:n_processes]
        work_completed = zeros(Int, n_processes)

        new{T}(
            n_processes,
            process_id,
            process_states,
            nothing,
            nothing,
            nothing,
            work_assignments,
            work_completed,
            zero(T),
            zero(T),
            Vector{T}(undef, n_params),
            Vector{T}(undef, n_observables),
            0,
            0.0,
        )
    end
end

"""
    initialize_threaded_sampler!(sampler::ThreadedSampler{T},
                                local_state_template, n_samples::Int) where T

Initialize threaded sampler with local state template.

C実装参考: vmcmain.c 1行目から803行目まで
"""
function initialize_threaded_sampler!(
    sampler::ThreadedSampler{T},
    local_state_template,
    n_samples::Int,
) where {T}
    # Create thread states
    sampler.thread_states = [deepcopy(local_state_template) for _ = 1:sampler.n_threads]

    # Initialize work queue
    sampler.work_queue = collect(1:n_samples)

    # Initialize results arrays
    fill!(sampler.energy_results, zero(T))
    fill!(sampler.gradient_results, zero(T))
    fill!(sampler.observable_results, zero(T))

    # Reset statistics
    sampler.total_samples = 0
    sampler.total_accepted = 0
    fill!(sampler.thread_times, 0.0)
end

"""
    threaded_sampling_step!(sampler::ThreadedSampler{T}, thread_id::Int,
                           rng::AbstractRNG) where T

Perform one sampling step on a specific thread.
"""
function threaded_sampling_step!(
    sampler::ThreadedSampler{T},
    thread_id::Int,
    rng::AbstractRNG,
) where {T}
    if thread_id > length(sampler.thread_states)
        return
    end

    state = sampler.thread_states[thread_id]

    # Get work item
    if isempty(sampler.work_queue)
        return
    end

    work_item = pop!(sampler.work_queue)

    # Perform sampling step
    start_time = time()
    enhanced_metropolis_step!(state, rng)
    end_time = time()

    # Update timing
    sampler.thread_times[thread_id] += end_time - start_time

    # Update statistics
    sampler.total_samples += 1
    if state.n_accepted > 0
        sampler.total_accepted += 1
    end
end

"""
    run_threaded_sampling!(sampler::ThreadedSampler{T}, n_steps::Int) where T

Run threaded sampling for specified number of steps.
"""
function run_threaded_sampling!(sampler::ThreadedSampler{T}, n_steps::Int) where {T}
    # Create thread-local RNGs
    rngs = [StableRNG(123 + i) for i = 1:sampler.n_threads]

    # Run sampling in parallel
    Threads.@threads for thread_id = 1:sampler.n_threads
        for _ = 1:n_steps
            threaded_sampling_step!(sampler, thread_id, rngs[thread_id])
        end
    end
end

"""
    initialize_distributed_sampler!(sampler::DistributedSampler{T},
                                   local_state_template, n_samples::Int) where T

Initialize distributed sampler with local state template.
"""
function initialize_distributed_sampler!(
    sampler::DistributedSampler{T},
    local_state_template,
    n_samples::Int,
) where {T}
    # Create process states
    sampler.process_states = [deepcopy(local_state_template) for _ = 1:sampler.n_processes]

    # Distribute work
    work_per_process = n_samples ÷ sampler.n_processes
    remaining_work = n_samples % sampler.n_processes

    for i = 1:sampler.n_processes
        start_idx = (i - 1) * work_per_process + 1
        end_idx = i * work_per_process
        if i <= remaining_work
            end_idx += 1
        end
        sampler.work_assignments[i] = collect(start_idx:end_idx)
    end

    # Initialize results
    fill!(sampler.global_gradients, zero(T))
    fill!(sampler.global_observables, zero(T))

    # Reset statistics
    sampler.total_samples = 0
    sampler.communication_overhead = 0.0
end

"""
    distributed_sampling_step!(sampler::DistributedSampler{T}, process_id::Int,
                              rng::AbstractRNG) where T

Perform one sampling step on a specific process.
"""
function distributed_sampling_step!(
    sampler::DistributedSampler{T},
    process_id::Int,
    rng::AbstractRNG,
) where {T}
    if process_id > length(sampler.process_states)
        return
    end

    state = sampler.process_states[process_id]

    # Check if work is available
    if sampler.work_completed[process_id] >= length(sampler.work_assignments[process_id])
        return
    end

    # Perform sampling step
    start_time = time()
    enhanced_metropolis_step!(state, rng)
    end_time = time()

    # Update work completion
    sampler.work_completed[process_id] += 1
    sampler.total_samples += 1
end

"""
    synchronize_distributed_results!(sampler::DistributedSampler{T}) where T

Synchronize results across all processes.
"""
function synchronize_distributed_results!(sampler::DistributedSampler{T}) where {T}
    start_time = time()

    # Collect energy results
    local_energies = [state.total_energy for state in sampler.process_states]
    sampler.global_energy_mean = mean(local_energies)
    sampler.global_energy_variance = var(local_energies)

    # Collect gradient results
    local_gradients =
        [get_rbm_parameters(state.rbm_network) for state in sampler.process_states]
    sampler.global_gradients = mean(local_gradients)

    # Collect observable results
    local_observables = [
        get_observable_statistics(state.observable_manager, "energy").mean for
        state in sampler.process_states
    ]
    sampler.global_observables = local_observables

    end_time = time()
    sampler.communication_overhead += end_time - start_time
end

"""
    run_distributed_sampling!(sampler::DistributedSampler{T}, n_steps::Int) where T

Run distributed sampling for specified number of steps.
"""
function run_distributed_sampling!(sampler::DistributedSampler{T}, n_steps::Int) where {T}
    # Create process-local RNGs
    rngs = [StableRNG(123 + i) for i = 1:sampler.n_processes]

    # Run sampling on each process
    for process_id = 1:sampler.n_processes
        for _ = 1:n_steps
            distributed_sampling_step!(sampler, process_id, rngs[process_id])
        end
    end

    # Synchronize results
    synchronize_distributed_results!(sampler)
end

"""
    ParallelSamplingManager{T}

Manages parallel sampling across different execution models.
"""
mutable struct ParallelSamplingManager{T<:Union{Float64,ComplexF64}}
    # Execution models
    threaded_sampler::Union{Nothing,ThreadedSampler{T}}
    distributed_sampler::Union{Nothing,DistributedSampler{T}}

    # Configuration
    config::ParallelConfig

    # Performance monitoring
    total_sampling_time::Float64
    total_communication_time::Float64
    total_synchronization_time::Float64

    # Results
    global_energy_mean::T
    global_energy_variance::T
    global_acceptance_rate::Float64

    function ParallelSamplingManager{T}(config::ParallelConfig) where {T}
        new{T}(nothing, nothing, config, 0.0, 0.0, 0.0, zero(T), zero(T), 0.0)
    end
end

"""
    initialize_parallel_sampling!(manager::ParallelSamplingManager{T},
                                 local_state_template, n_samples::Int) where T

Initialize parallel sampling manager.
"""
function initialize_parallel_sampling!(
    manager::ParallelSamplingManager{T},
    local_state_template,
    n_samples::Int,
) where {T}
    if manager.config.use_threading
        manager.threaded_sampler =
            ThreadedSampler{T}(manager.config.n_threads, n_samples, 100, 50)
        initialize_threaded_sampler!(
            manager.threaded_sampler,
            local_state_template,
            n_samples,
        )
    end

    if manager.config.use_distributed
        manager.distributed_sampler =
            DistributedSampler{T}(manager.config.n_processes, 1, n_samples, 100, 50)
        initialize_distributed_sampler!(
            manager.distributed_sampler,
            local_state_template,
            n_samples,
        )
    end
end

"""
    run_parallel_sampling!(manager::ParallelSamplingManager{T}, n_steps::Int) where T

Run parallel sampling for specified number of steps.
"""
function run_parallel_sampling!(manager::ParallelSamplingManager{T}, n_steps::Int) where {T}
    start_time = time()

    if manager.config.use_threading && manager.threaded_sampler !== nothing
        run_threaded_sampling!(manager.threaded_sampler, n_steps)
    end

    if manager.config.use_distributed && manager.distributed_sampler !== nothing
        run_distributed_sampling!(manager.distributed_sampler, n_steps)
    end

    end_time = time()
    manager.total_sampling_time += end_time - start_time

    # Aggregate results
    aggregate_parallel_results!(manager)
end

"""
    aggregate_parallel_results!(manager::ParallelSamplingManager{T}) where T

Aggregate results from all parallel execution models.
"""
function aggregate_parallel_results!(manager::ParallelSamplingManager{T}) where {T}
    energies = T[]
    acceptance_rates = Float64[]

    # Collect from threaded sampler
    if manager.threaded_sampler !== nothing
        for state in manager.threaded_sampler.thread_states
            push!(energies, state.total_energy)
            push!(acceptance_rates, state.n_accepted / max(state.n_updates, 1))
        end
    end

    # Collect from distributed sampler
    if manager.distributed_sampler !== nothing
        push!(energies, manager.distributed_sampler.global_energy_mean)
        push!(
            acceptance_rates,
            manager.distributed_sampler.global_energy_mean /
            max(manager.distributed_sampler.total_samples, 1),
        )
    end

    # Aggregate results
    if !isempty(energies)
        manager.global_energy_mean = mean(energies)
        manager.global_energy_variance = var(energies)
        manager.global_acceptance_rate = mean(acceptance_rates)
    end
end

"""
    get_parallel_statistics(manager::ParallelSamplingManager{T}) where T

Get comprehensive parallel sampling statistics.
"""
function get_parallel_statistics(manager::ParallelSamplingManager{T}) where {T}
    stats = Dict{String,Any}()

    # Basic statistics
    stats["total_sampling_time"] = manager.total_sampling_time
    stats["total_communication_time"] = manager.total_communication_time
    stats["total_synchronization_time"] = manager.total_synchronization_time
    stats["global_energy_mean"] = manager.global_energy_mean
    stats["global_energy_variance"] = manager.global_energy_variance
    stats["global_acceptance_rate"] = manager.global_acceptance_rate

    # Threaded sampler statistics
    if manager.threaded_sampler !== nothing
        stats["threaded_total_samples"] = manager.threaded_sampler.total_samples
        stats["threaded_total_accepted"] = manager.threaded_sampler.total_accepted
        stats["threaded_acceptance_rate"] =
            manager.threaded_sampler.total_accepted /
            max(manager.threaded_sampler.total_samples, 1)
        stats["thread_times"] = manager.threaded_sampler.thread_times
    end

    # Distributed sampler statistics
    if manager.distributed_sampler !== nothing
        stats["distributed_total_samples"] = manager.distributed_sampler.total_samples
        stats["distributed_communication_overhead"] =
            manager.distributed_sampler.communication_overhead
        stats["work_completed_per_process"] = manager.distributed_sampler.work_completed
    end

    return stats
end

"""
    benchmark_parallel_sampling(n_samples::Int = 10000, n_steps::Int = 100)

Benchmark parallel sampling performance.
"""
function benchmark_parallel_sampling(n_samples::Int = 10000, n_steps::Int = 100)
    println("Benchmarking parallel sampling (n_samples=$n_samples, n_steps=$n_steps)...")

    # Test threaded sampling
    config_threaded = ParallelConfig(
        use_threading = true,
        use_distributed = false,
        n_threads = Threads.nthreads(),
    )
    manager_threaded = ParallelSamplingManager{ComplexF64}(config_threaded)

    # Create dummy local state template
    local_state_template = EnhancedVMCState{ComplexF64}(5, 10)

    # Initialize and run threaded sampling
    initialize_parallel_sampling!(manager_threaded, local_state_template, n_samples)

    @time begin
        run_parallel_sampling!(manager_threaded, n_steps)
    end

    # Get statistics
    stats = get_parallel_statistics(manager_threaded)
    println("Threaded sampling completed.")
    println("  Total samples: $(stats["threaded_total_samples"])")
    println("  Acceptance rate: $(stats["threaded_acceptance_rate"])")
    println("  Sampling time: $(stats["total_sampling_time"])")

    # Test distributed sampling (if available)
    if nprocs() > 1
        config_distributed = ParallelConfig(
            use_threading = false,
            use_distributed = true,
            n_processes = nprocs(),
        )
        manager_distributed = ParallelSamplingManager{ComplexF64}(config_distributed)

        initialize_parallel_sampling!(manager_distributed, local_state_template, n_samples)

        @time begin
            run_parallel_sampling!(manager_distributed, n_steps)
        end

        stats_distributed = get_parallel_statistics(manager_distributed)
        println("Distributed sampling completed.")
        println("  Total samples: $(stats_distributed["distributed_total_samples"])")
        println(
            "  Communication overhead: $(stats_distributed["distributed_communication_overhead"])",
        )
    end

    println("Parallel sampling benchmark completed.")
end
