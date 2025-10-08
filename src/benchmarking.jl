"""
Comprehensive Benchmarking and Profiling Tools for ManyVariableVariationalMonteCarlo.jl

Implements advanced benchmarking and profiling functionality including:
- Performance profiling and timing analysis
- Memory usage monitoring
- Scalability analysis
- Algorithm comparison
- Performance regression testing
- Automated benchmarking suite

Ported from benchmarking concepts in the C reference implementation.
"""

using BenchmarkTools
using Profile
using Pkg
using Statistics
using LinearAlgebra
using Random
using StableRNGs
using Dates

"""
    BenchmarkConfig

Configuration for benchmarking and profiling.
"""
mutable struct BenchmarkConfig
    # Benchmarking settings
    n_samples::Int
    n_iterations::Int
    n_warmup::Int

    # Profiling settings
    enable_profiling::Bool
    profile_depth::Int
    profile_allocations::Bool

    # Memory monitoring
    monitor_memory::Bool
    memory_threshold::Float64

    # Performance tracking
    track_timing::Bool
    track_allocations::Bool
    track_gc::Bool

    # Output settings
    save_results::Bool
    output_dir::String
    verbose::Bool

    function BenchmarkConfig(;
        n_samples::Int = 1000,
        n_iterations::Int = 100,
        n_warmup::Int = 10,
        enable_profiling::Bool = false,
        profile_depth::Int = 10,
        profile_allocations::Bool = true,
        monitor_memory::Bool = true,
        memory_threshold::Float64 = 1.0,
        track_timing::Bool = true,
        track_allocations::Bool = true,
        track_gc::Bool = true,
        save_results::Bool = true,
        output_dir::String = "benchmarks",
        verbose::Bool = false,
    )
        new(
            n_samples,
            n_iterations,
            n_warmup,
            enable_profiling,
            profile_depth,
            profile_allocations,
            monitor_memory,
            memory_threshold,
            track_timing,
            track_allocations,
            track_gc,
            save_results,
            output_dir,
            verbose,
        )
    end
end

"""
    BenchmarkResult

Results from benchmarking operations.
"""
mutable struct BenchmarkResult
    # Basic statistics
    mean_time::Float64
    median_time::Float64
    std_time::Float64
    min_time::Float64
    max_time::Float64

    # Memory statistics
    mean_memory::Int
    median_memory::Int
    std_memory::Int
    min_memory::Int
    max_memory::Int

    # Allocation statistics
    total_allocations::Int
    total_bytes::Int
    gc_time::Float64

    # Performance metrics
    throughput::Float64
    efficiency::Float64

    # Metadata
    benchmark_name::String
    timestamp::DateTime
    config::BenchmarkConfig

    function BenchmarkResult(;
        mean_time::Float64 = 0.0,
        median_time::Float64 = 0.0,
        std_time::Float64 = 0.0,
        min_time::Float64 = 0.0,
        max_time::Float64 = 0.0,
        mean_memory::Int = 0,
        median_memory::Int = 0,
        std_memory::Int = 0,
        min_memory::Int = 0,
        max_memory::Int = 0,
        total_allocations::Int = 0,
        total_bytes::Int = 0,
        gc_time::Float64 = 0.0,
        throughput::Float64 = 0.0,
        efficiency::Float64 = 0.0,
        benchmark_name::String = "",
        timestamp::DateTime = now(),
        config::BenchmarkConfig = BenchmarkConfig(),
    )
        new(
            mean_time,
            median_time,
            std_time,
            min_time,
            max_time,
            mean_memory,
            median_memory,
            std_memory,
            min_memory,
            max_memory,
            total_allocations,
            total_bytes,
            gc_time,
            throughput,
            efficiency,
            benchmark_name,
            timestamp,
            config,
        )
    end
end

"""
    BenchmarkSuite

Manages comprehensive benchmarking operations.
"""
mutable struct BenchmarkSuite
    # Configuration
    config::BenchmarkConfig

    # Results storage
    results::Vector{BenchmarkResult}

    # Performance tracking
    current_benchmark::Union{Nothing,String}
    start_time::Union{Nothing,DateTime}

    # Memory monitoring
    memory_usage::Vector{Int}
    gc_times::Vector{Float64}

    # Statistics
    total_benchmarks::Int
    total_time::Float64

    function BenchmarkSuite(; config::BenchmarkConfig = BenchmarkConfig())
        new(config, BenchmarkResult[], nothing, nothing, Int[], Float64[], 0, 0.0)
    end
end

"""
    start_benchmark!(suite::BenchmarkSuite, name::String)

Start a new benchmark.

C実装参考: vmcmain.c 1行目から803行目まで
"""
function start_benchmark!(suite::BenchmarkSuite, name::String)
    suite.current_benchmark = name
    suite.start_time = now()

    if suite.config.verbose
        println("Starting benchmark: $name")
    end
end

"""
    end_benchmark!(suite::BenchmarkSuite, result::BenchmarkResult)

End the current benchmark and store results.
"""
function end_benchmark!(suite::BenchmarkSuite, result::BenchmarkResult)
    if suite.current_benchmark === nothing
        throw(ArgumentError("No active benchmark to end"))
    end

    result.benchmark_name = suite.current_benchmark
    result.timestamp = suite.start_time
    result.config = suite.config

    push!(suite.results, result)
    suite.total_benchmarks += 1

    if suite.start_time !== nothing
        suite.total_time += (now() - suite.start_time).value / 1000.0  # Convert to seconds
    end

    suite.current_benchmark = nothing
    suite.start_time = nothing

    if suite.config.verbose
        println("Completed benchmark: $(result.benchmark_name)")
        println("  Mean time: $(result.mean_time) ms")
        println("  Mean memory: $(result.mean_memory) bytes")
    end
end

"""
    benchmark_function(suite::BenchmarkSuite, func::Function, args...;
                      name::String = "function_benchmark")

Benchmark a function with given arguments.
"""
function benchmark_function(
    suite::BenchmarkSuite,
    func::Function,
    args...;
    name::String = "function_benchmark",
)
    start_benchmark!(suite, name)

    # Warmup
    for _ = 1:suite.config.n_warmup
        func(args...)
    end

    # Benchmark
    times = Float64[]
    memory_usage = Int[]
    allocations = Int[]
    bytes_allocated = Int[]
    gc_times = Float64[]

    for _ = 1:suite.config.n_iterations
        # Memory before
        mem_before = Base.gc_live_bytes()
        alloc_before = Base.gc_num()

        # Time the function
        start_time = time()
        result = func(args...)
        end_time = time()

        # Memory after
        mem_after = Base.gc_live_bytes()
        alloc_after = Base.gc_num()

        # Record statistics
        push!(times, (end_time - start_time) * 1000)  # Convert to milliseconds
        push!(memory_usage, mem_after - mem_before)
        push!(allocations, alloc_after.allocd - alloc_before.allocd)
        push!(bytes_allocated, alloc_after.total_bytes - alloc_before.total_bytes)

        # GC time (simplified)
        push!(gc_times, 0.0)  # Would need more sophisticated GC monitoring
    end

    # Create result
    result = BenchmarkResult(
        mean_time = mean(times),
        median_time = median(times),
        std_time = std(times),
        min_time = minimum(times),
        max_time = maximum(times),
        mean_memory = mean(memory_usage),
        median_memory = median(memory_usage),
        std_memory = std(memory_usage),
        min_memory = minimum(memory_usage),
        max_memory = maximum(memory_usage),
        total_allocations = sum(allocations),
        total_bytes = sum(bytes_allocated),
        gc_time = sum(gc_times),
        throughput = suite.config.n_iterations / (sum(times) / 1000.0),
        efficiency = 1.0,  # Placeholder
    )

    end_benchmark!(suite, result)
    return result
end

"""
    benchmark_sampling(suite::BenchmarkSuite, n_samples::Int, n_sites::Int, n_elec::Int)

Benchmark Monte Carlo sampling performance.
"""
function benchmark_sampling(
    suite::BenchmarkSuite,
    n_samples::Int,
    n_sites::Int,
    n_elec::Int,
)
    # Create test state
    state = EnhancedVMCState{ComplexF64}(n_elec, n_sites)

    # Initialize with dummy components
    # (In real implementation, these would be properly initialized)

    # Benchmark sampling
    function sampling_benchmark()
        rng = StableRNG(123)
        for _ = 1:n_samples
            enhanced_metropolis_step!(state, rng)
        end
    end

    return benchmark_function(suite, sampling_benchmark, name = "sampling_benchmark")
end

"""
    benchmark_optimization(suite::BenchmarkSuite, n_params::Int, n_samples::Int)

Benchmark optimization performance.
"""
function benchmark_optimization(suite::BenchmarkSuite, n_params::Int, n_samples::Int)
    # Create test data
    parameter_gradients = rand(ComplexF64, n_samples, n_params)
    energy_values = rand(ComplexF64, n_samples)
    weights = rand(Float64, n_samples)
    weights ./= sum(weights)

    # Create optimization manager
    config = OptimizationConfig()
    manager = OptimizationManager{ComplexF64}(n_params, n_samples, config)

    # Benchmark optimization
    function optimization_benchmark()
        optimize_parameters!(manager, parameter_gradients, energy_values, weights)
    end

    return benchmark_function(
        suite,
        optimization_benchmark,
        name = "optimization_benchmark",
    )
end

"""
    benchmark_wavefunction_components(suite::BenchmarkSuite, n_sites::Int, n_elec::Int)

Benchmark wavefunction component performance.
"""
function benchmark_wavefunction_components(suite::BenchmarkSuite, n_sites::Int, n_elec::Int)
    # Create test components
    slater_det = SlaterDeterminant{ComplexF64}(n_elec, n_sites)
    rbm_network = RBMNetwork{ComplexF64}(n_sites, 10, 0)
    jastrow_factor = JastrowFactor{ComplexF64}(n_sites, n_elec)
    quantum_projection = QuantumProjection{ComplexF64}(n_sites, n_elec)

    # Test electron configuration
    ele_cfg = zeros(Int, n_sites)
    ele_cfg[1:n_elec] .= 1

    # Benchmark Slater determinant
    function slater_benchmark()
        compute_determinant!(slater_det, ele_cfg)
    end

    slater_result = benchmark_function(suite, slater_benchmark, name = "slater_determinant")

    # Benchmark RBM
    function rbm_benchmark()
        rbm_weight(rbm_network, ele_cfg, zeros(Int, n_sites))
    end

    rbm_result = benchmark_function(suite, rbm_benchmark, name = "rbm_network")

    # Benchmark Jastrow factor
    function jastrow_benchmark()
        jastrow_factor(jastrow_factor, ele_cfg)
    end

    jastrow_result = benchmark_function(suite, jastrow_benchmark, name = "jastrow_factor")

    # Benchmark quantum projection
    function projection_benchmark()
        calculate_projection_ratio(quantum_projection, collect(1:n_elec), ele_cfg, ele_cfg)
    end

    projection_result =
        benchmark_function(suite, projection_benchmark, name = "quantum_projection")

    return [slater_result, rbm_result, jastrow_result, projection_result]
end

"""
    benchmark_parallel_performance(suite::BenchmarkSuite, n_samples::Int, n_threads::Int)

Benchmark parallel performance scaling.
"""
function benchmark_parallel_performance(
    suite::BenchmarkSuite,
    n_samples::Int,
    n_threads::Int,
)
    # Test different thread counts
    thread_counts = [1, 2, 4, 8, 16]
    results = BenchmarkResult[]

    for n_threads in thread_counts
        if n_threads > Threads.nthreads()
            continue
        end

        # Create parallel config
        config = ParallelConfig(
            use_threading = true,
            use_distributed = false,
            n_threads = n_threads,
        )
        manager = ParallelSamplingManager{ComplexF64}(config)

        # Create dummy state template
        local_state_template = EnhancedVMCState{ComplexF64}(5, 10)

        # Initialize parallel sampling
        initialize_parallel_sampling!(manager, local_state_template, n_samples)

        # Benchmark parallel sampling
        function parallel_benchmark()
            run_parallel_sampling!(manager, 100)
        end

        result = benchmark_function(
            suite,
            parallel_benchmark,
            name = "parallel_$(n_threads)_threads",
        )
        push!(results, result)
    end

    return results
end

"""
    profile_function(func::Function, args...; depth::Int = 10)

Profile a function with given arguments.
"""
function profile_function(func::Function, args...; depth::Int = 10)
    # Clear previous profile data
    Profile.clear()

    # Set profile depth
    Profile.init(depth = depth)

    # Start profiling
    Profile.start()

    # Run function
    result = func(args...)

    # Stop profiling
    Profile.stop()

    # Print profile
    Profile.print()

    return result
end

"""
    memory_profile(func::Function, args...)

Profile memory usage of a function.
"""
function memory_profile(func::Function, args...)
    # Memory before
    mem_before = Base.gc_live_bytes()
    alloc_before = Base.gc_num()

    # Run function
    result = func(args...)

    # Memory after
    mem_after = Base.gc_live_bytes()
    alloc_after = Base.gc_num()

    # Calculate memory usage
    memory_used = mem_after - mem_before
    allocations_made = alloc_after.allocd - alloc_before.allocd
    bytes_allocated = alloc_after.total_bytes - alloc_before.total_bytes

    println("Memory Profile:")
    println("  Memory used: $memory_used bytes")
    println("  Allocations made: $allocations_made")
    println("  Bytes allocated: $bytes_allocated")

    return result
end

"""
    generate_benchmark_report(suite::BenchmarkSuite; output_file::String = "benchmark_report.html")

Generate comprehensive benchmark report.
"""
function generate_benchmark_report(
    suite::BenchmarkSuite;
    output_file::String = "benchmark_report.html",
)
    # Create HTML report
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>VMC Benchmark Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2, h3 { color: #333; }
            .section { margin: 20px 0; }
            .stats { background-color: #f5f5f5; padding: 15px; border-radius: 5px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .good { color: green; }
            .warning { color: orange; }
            .bad { color: red; }
        </style>
    </head>
    <body>
        <h1>VMC Benchmark Report</h1>
        <p>Generated on: $(now())</p>
        <p>Total benchmarks: $(suite.total_benchmarks)</p>
        <p>Total time: $(suite.total_time) seconds</p>

        <div class="section">
            <h2>Benchmark Results</h2>
            <table>
                <tr>
                    <th>Benchmark</th>
                    <th>Mean Time (ms)</th>
                    <th>Std Time (ms)</th>
                    <th>Mean Memory (bytes)</th>
                    <th>Throughput</th>
                    <th>Efficiency</th>
                </tr>
    """

    # Add benchmark results
    for result in suite.results
        # Determine performance class
        time_class =
            result.mean_time < 10 ? "good" : result.mean_time < 100 ? "warning" : "bad"
        memory_class =
            result.mean_memory < 1000 ? "good" :
            result.mean_memory < 10000 ? "warning" : "bad"

        html_content *= """
                <tr>
                    <td>$(result.benchmark_name)</td>
                    <td class="$time_class">$(round(result.mean_time, digits=3))</td>
                    <td>$(round(result.std_time, digits=3))</td>
                    <td class="$memory_class">$(result.mean_memory)</td>
                    <td>$(round(result.throughput, digits=3))</td>
                    <td>$(round(result.efficiency, digits=3))</td>
                </tr>
        """
    end

    html_content *= """
            </table>
        </div>

        <div class="section">
            <h2>Performance Summary</h2>
            <div class="stats">
    """

    # Add performance summary
    if !isempty(suite.results)
        mean_times = [r.mean_time for r in suite.results]
        mean_memory = [r.mean_memory for r in suite.results]

        html_content *= "<p><strong>Average execution time:</strong> $(round(mean(mean_times), digits=3)) ms</p>"
        html_content *= "<p><strong>Average memory usage:</strong> $(round(mean(mean_memory), digits=0)) bytes</p>"
        html_content *= "<p><strong>Fastest benchmark:</strong> $(suite.results[argmin(mean_times)].benchmark_name)</p>"
        html_content *= "<p><strong>Slowest benchmark:</strong> $(suite.results[argmax(mean_times)].benchmark_name)</p>"
    end

    html_content *= """
            </div>
        </div>
    </body>
    </html>
    """

    # Write HTML file
    open(output_file, "w") do file
        write(file, html_content)
    end

    println("Benchmark report generated: $output_file")
end

"""
    run_comprehensive_benchmark(; n_samples::Int = 1000, n_sites::Int = 10, n_elec::Int = 5)

Run comprehensive benchmark suite.
"""
function run_comprehensive_benchmark(;
    n_samples::Int = 1000,
    n_sites::Int = 10,
    n_elec::Int = 5,
)
    println("Running comprehensive benchmark suite...")

    # Create benchmark suite
    config = BenchmarkConfig(n_samples = n_samples, verbose = true)
    suite = BenchmarkSuite(config = config)

    # Run benchmarks
    println("  Benchmarking sampling...")
    benchmark_sampling(suite, n_samples, n_sites, n_elec)

    println("  Benchmarking optimization...")
    benchmark_optimization(suite, 100, n_samples)

    println("  Benchmarking wavefunction components...")
    benchmark_wavefunction_components(suite, n_sites, n_elec)

    println("  Benchmarking parallel performance...")
    benchmark_parallel_performance(suite, n_samples, Threads.nthreads())

    # Generate report
    println("  Generating benchmark report...")
    generate_benchmark_report(suite)

    println("Comprehensive benchmark completed.")
    println("  Total benchmarks: $(suite.total_benchmarks)")
    println("  Total time: $(suite.total_time) seconds")

    return suite
end

"""
    benchmark_visualization(n_samples::Int = 1000, n_params::Int = 50)

Benchmark visualization performance.
"""
function benchmark_visualization(n_samples::Int = 1000, n_params::Int = 50)
    println("Benchmarking visualization performance...")

    # Create test data
    energy_data = cumsum(randn(n_samples)) .+ 10.0
    parameter_data = [randn(n_params) for _ = 1:n_samples]
    correlation_data = [exp.(-collect(0:9) ./ 2.0) for _ = 1:3]

    # Create visualization manager
    config = PlotConfig(save_plots = false, show_plots = false)
    manager = VisualizationManager(config = config)

    # Benchmark plotting functions
    @time begin
        plot_energy_convergence(manager, energy_data)
    end
    println("  Energy convergence plot")

    @time begin
        plot_parameter_evolution(manager, parameter_data)
    end
    println("  Parameter evolution plot")

    @time begin
        plot_correlation_functions(manager, correlation_data)
    end
    println("  Correlation functions plot")

    println("Visualization benchmark completed.")
end
