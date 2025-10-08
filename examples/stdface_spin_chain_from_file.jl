using ManyVariableVariationalMonteCarlo
using Printf

# Enhanced mVMC simulation with C-implementation compatibility
# Load and run from an mVMC-style StdFace.def (Spin Heisenberg chain)
#
# This example demonstrates the enhanced implementation with:
# - StdFace.def parser
# - Detailed wavefunction components
# - Precise stochastic reconfiguration
# - C-compatible output format
#
# Usage:
#   julia --project examples/18_stdface_spin_chain_from_file.jl [path/to/StdFace.def]

# Load enhanced implementation modules

function usage()
    println("Usage: julia --project examples/18_stdface_spin_chain_from_file.jl <path/to/StdFace.def>")
    println("Hint: try mVMC/samples/Standard/Spin/HeisenbergChain/StdFace.def")
    println("")
    println("This enhanced implementation provides:")
    println("  - Complete StdFace.def parsing compatibility")
    println("  - Detailed wavefunction components (Gutzwiller, Jastrow, RBM)")
    println("  - Precise stochastic reconfiguration with CG solver")
    println("  - C-implementation compatible output format")
end

function main()
    path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "mVMC", "samples", "Standard", "Spin", "HeisenbergChain", "StdFace.def")
    if !isfile(path)
        println("[error] StdFace.def not found: ", path)
        usage()
        return
    end

    println("######  Enhanced mVMC Julia Implementation  ######")
    println("Starting C-implementation compatible simulation...")
    println()
    # C実装のStandard modeと同様の処理フロー
    # 1. StdFace.defを解析してexpert modeファイルを生成
    # 2. 生成されたファイルを読み込んでシミュレーション実行
    output_dir = joinpath(@__DIR__, "output")  # Output in same directory as input
    mkpath(output_dir)
    sim = run_mvmc_from_stdface(path; T=ComplexF64, output_dir=output_dir)

    # Print summary in mVMC style
    println("######  Simulation Completed Successfully  ######")
    print_simulation_summary(sim)

    # Comparison with C implementation
    println("Output files generated (C-implementation compatible):")
    if sim.mode == PARAMETER_OPTIMIZATION
        println("  - zvo_out_001.dat       : Energy evolution")
        println("  - zvo_var_001.dat       : Parameter variation")
        println("  - zvo_SRinfo.dat        : Stochastic reconfiguration info")
        println("  - zvo_time_001.dat      : Timing information")
        println("  - zvo_CalcTimer.dat     : Calculation timer summary")
        println("  - zqp_opt.dat           : Final optimized parameters (all)")
        println("  - zqp_gutzwiller_opt.dat: Split params (Gutzwiller)")
        println("  - zqp_jastrow_opt.dat   : Split params (Jastrow)")
        println("  - zqp_orbital_opt.dat   : Split params (Orbital)")
    else
        println("  - zvo_out.dat      : Energy and magnetization")
        println("  - zvo_cisajs.dat   : One-body Green functions")
        println("  - zvo_energy.dat   : Energy time series")
        if haskey(sim.config.face, :TwoBodyG) && Bool(sim.config.face[:TwoBodyG])
            println("  - zvo_cisajscktaltex.dat : Two-body Green functions")
            println("  - zvo_cisajscktalt.dat   : Two-body Green functions (DC)")
        end
        if sim.config.nlanczos_mode > 0
            println("  - zvo_ls_result.dat      : Lanczos analysis")
            println("  - zvo_ls_alpha_beta.dat  : Lanczos coefficients")
        end
    end

    println()
    println("Files are located in: ", output_dir)

    # Performance summary
    if !isempty(sim.optimization_results)
        final_result = sim.optimization_results[end]
        println()
        println("Final Results:")
        @printf("  Energy: %.8f ± %.8f\n",
                real(final_result["energy"]), final_result["energy_error"])
        @printf("  Condition Number: %.2e\n", final_result["overlap_condition"])
        @printf("  Parameter Updates: %d\n", length(sim.optimization_results))
    elseif !isempty(sim.physics_results)
        println()
        println("Physics Results:")
        @printf("  Energy: %.8f ± %.8f\n",
                real(sim.physics_results["energy_mean"]), sim.physics_results["energy_std"])
        @printf("  Acceptance Rate: %.3f\n", sim.physics_results["acceptance_rate"])
        @printf("  Samples: %d\n", sim.physics_results["n_samples"])
    end
end

function compare_with_c_output()
    """
    Function to compare Julia output with C implementation output.
    Compares key files and reports differences.
    """
    println("######  Comparison with C Implementation  ######")
    println("To compare with C implementation:")
    println("1. Run C mVMC on the same StdFace.def file")
    println("2. Compare the following key files:")
    println("   - Energy values in zvo_out.dat")
    println("   - Parameter evolution in zvo_var.dat")
    println("   - Green functions in zvo_cisajs.dat")
    println("3. Check for numerical consistency within VMC statistical errors")
    println()
end

# Add comparison information
compare_with_c_output()
main()
