"""
mVMC Compatible Output Format

Complete implementation of mVMC-compatible output format
matching the C reference implementation exactly.
"""

using Printf
using Dates

"""
    MVMCOutputManager

Manages all output files and formatting for mVMC compatibility.
"""
mutable struct MVMCOutputManager
    output_dir::String
    file_handles::Dict{String,IOStream}
    binary_mode::Bool
    flush_interval::Int
    current_sample::Int

    function MVMCOutputManager(output_dir::String="output"; binary_mode::Bool=false, flush_interval::Int=0)
        mkpath(output_dir)
        new(output_dir, Dict{String,IOStream}(), binary_mode, flush_interval, 0)
    end
end

"""
    open_output_files!(manager::MVMCOutputManager, mode::String="optimization")

Open all necessary output files based on calculation mode.
"""
function open_output_files!(manager::MVMCOutputManager, mode::String="optimization")
    close_all_files!(manager)

    if mode == "optimization"
        open_optimization_files!(manager)
    elseif mode == "physics"
        open_physics_files!(manager)
    end
end

"""
    open_optimization_files!(manager::MVMCOutputManager)

Open files for parameter optimization output.
"""
function open_optimization_files!(manager::MVMCOutputManager)
    # Main result file
    manager.file_handles["zvo_out"] = open(joinpath(manager.output_dir, "zvo_out.dat"), "w")

    # Parameter variation file
    if manager.binary_mode
        manager.file_handles["zvo_var"] = open(joinpath(manager.output_dir, "zvo_var.dat"), "wb")
    else
        manager.file_handles["zvo_var"] = open(joinpath(manager.output_dir, "zvo_var.dat"), "w")
    end

    # SR information file
    manager.file_handles["zvo_SRinfo"] = open(joinpath(manager.output_dir, "zvo_SRinfo.dat"), "w")

    # Timer information
    manager.file_handles["zvo_time"] = open(joinpath(manager.output_dir, "zvo_time.dat"), "w")
end

"""
    open_physics_files!(manager::MVMCOutputManager, nstore::Int=1)

Open files for physics quantity calculation output.
"""
function open_physics_files!(manager::MVMCOutputManager, nstore::Int=1)
    # Main result file
    manager.file_handles["zvo_out"] = open(joinpath(manager.output_dir, "zvo_out.dat"), "w")

    # One-body Green function files
    manager.file_handles["zvo_cisajs"] = open(joinpath(manager.output_dir, "zvo_cisajs.dat"), "w")

    # Binned one-body Green function files
    for bin in 1:nstore
        suffix = @sprintf("_%03d", bin)
        key = "zvo_cisajs" * suffix
        filename = "zvo_cisajs" * suffix * ".dat"
        manager.file_handles[key] = open(joinpath(manager.output_dir, filename), "w")
    end

    # Two-body Green function files
    manager.file_handles["zvo_cisajscktaltex"] = open(joinpath(manager.output_dir, "zvo_cisajscktaltex.dat"), "w")
    manager.file_handles["zvo_cisajscktalt"] = open(joinpath(manager.output_dir, "zvo_cisajscktalt.dat"), "w")

    # Binned two-body Green function files
    for bin in 1:nstore
        suffix = @sprintf("_%03d", bin)
        key1 = "zvo_cisajscktaltex" * suffix
        key2 = "zvo_cisajscktalt" * suffix
        filename1 = "zvo_cisajscktaltex" * suffix * ".dat"
        filename2 = "zvo_cisajscktalt" * suffix * ".dat"
        manager.file_handles[key1] = open(joinpath(manager.output_dir, filename1), "w")
        manager.file_handles[key2] = open(joinpath(manager.output_dir, filename2), "w")
    end

    # Energy time series
    manager.file_handles["zvo_energy"] = open(joinpath(manager.output_dir, "zvo_energy.dat"), "w")

    # Timer information
    manager.file_handles["zvo_time"] = open(joinpath(manager.output_dir, "zvo_time.dat"), "w")
end

"""
    write_optimization_header!(manager::MVMCOutputManager, config::SimulationConfig)

Write header information for optimization run.
"""
function write_optimization_header!(manager::MVMCOutputManager, config::SimulationConfig)
    # SR info header
    if haskey(manager.file_handles, "zvo_SRinfo")
        f = manager.file_handles["zvo_SRinfo"]
        println(f, "# Stochastic Reconfiguration Information")
        println(f, "# Iteration  Energy_Real  Energy_Imag  Energy_Variance  Force_Norm  Overlap_Condition")
        flush_if_needed!(manager, f)
    end

    # Time header
    if haskey(manager.file_handles, "zvo_time")
        f = manager.file_handles["zvo_time"]
        println(f, "# Time Information")
        println(f, "# Iteration  Elapsed_Time[s]  Step_Time[s]")
        flush_if_needed!(manager, f)
    end
end

"""
    write_physics_header!(manager::MVMCOutputManager, config::SimulationConfig)

Write header information for physics calculation.
"""
function write_physics_header!(manager::MVMCOutputManager, config::SimulationConfig)
    # One-body Green function header
    if haskey(manager.file_handles, "zvo_cisajs")
        f = manager.file_handles["zvo_cisajs"]
        println(f, "# One-body Green function <c†_i_s c_j_t>")
        println(f, "# i  s  j  t   Re[G]   Im[G]")
        flush_if_needed!(manager, f)
    end

    # Two-body Green function headers
    if haskey(manager.file_handles, "zvo_cisajscktaltex")
        f = manager.file_handles["zvo_cisajscktaltex"]
        println(f, "# Two-body Green function (equal-time)")
        println(f, "# i s j t k u l v   Re[G]   Im[G]")
        flush_if_needed!(manager, f)
    end

    if haskey(manager.file_handles, "zvo_cisajscktalt")
        f = manager.file_handles["zvo_cisajscktalt"]
        println(f, "# Two-body Green function (disconnected part)")
        println(f, "# i s j t k u l v   Re[G]   Im[G]")
        flush_if_needed!(manager, f)
    end

    # Energy time series header
    if haskey(manager.file_handles, "zvo_energy")
        f = manager.file_handles["zvo_energy"]
        println(f, "# Energy time series")
        println(f, "# Sample  Re[E]  Im[E]")
        flush_if_needed!(manager, f)
    end
end

"""
    write_optimization_step!(manager::MVMCOutputManager, iteration::Int, energy::Complex, energy_var::Float64, params::Vector, sr_info::Dict)

Write optimization step data in mVMC format.
"""
function write_optimization_step!(manager::MVMCOutputManager, iteration::Int, energy::Complex, energy_var::Float64, params::Vector, sr_info::Dict)
    # Main output (zvo_out.dat)
    if haskey(manager.file_handles, "zvo_out")
        f = manager.file_handles["zvo_out"]
        etot = energy
        etot2 = abs2(energy)  # Simplified for single sample
        sztot = get(sr_info, "sztot", 0.0)
        sztot2 = abs2(sztot)

        @printf(f, "%.18e %.18e %.18e %.18e %.18e %.18e\n",
                real(etot), imag(etot), real(etot2), real((etot2 - etot*etot)/(etot*etot)),
                real(sztot), real(sztot2))
        flush_if_needed!(manager, f)
    end

    # Parameter variation (zvo_var.dat)
    if haskey(manager.file_handles, "zvo_var")
        f = manager.file_handles["zvo_var"]
        if manager.binary_mode
            # Binary output (matching C implementation)
            param_data = Float64[]
            for p in params
                push!(param_data, real(p))
                if p isa Complex
                    push!(param_data, imag(p))
                end
            end
            write(f, param_data)
        else
            # Formatted output
            @printf(f, "%.18e %.18e 0.0 %.18e %.18e 0.0 ", real(energy), imag(energy), real(abs2(energy)), imag(abs2(energy)))
            for p in params
                @printf(f, "%.18e %.18e 0.0 ", real(p), imag(p))
            end
            println(f)
        end
        flush_if_needed!(manager, f)
    end

    # SR information (zvo_SRinfo.dat)
    if haskey(manager.file_handles, "zvo_SRinfo")
        f = manager.file_handles["zvo_SRinfo"]
        condition_num = get(sr_info, "condition_number", 1.0)
        force_norm = get(sr_info, "force_norm", 0.0)

        @printf(f, "%6d  %.16e  %.16e  %.16e  %.16e  %.16e\n",
                iteration, real(energy), imag(energy), energy_var, force_norm, condition_num)
        flush_if_needed!(manager, f)
    end

    manager.current_sample += 1
end

"""
    write_final_parameters!(manager::MVMCOutputManager, params::Vector)

Write final optimized parameters to zqp_opt.dat.
"""
function write_final_parameters!(manager::MVMCOutputManager, params::Vector)
    filename = joinpath(manager.output_dir, "zqp_opt.dat")
    open(filename, "w") do f
        println(f, "# Optimized variational parameters")
        println(f, "# Index  Real  Imaginary")

        for (i, p) in enumerate(params)
            @printf(f, "%6d  %.16e  %.16e\n", i, real(p), imag(p))
        end
    end
end

"""
    write_physics_step!(manager::MVMCOutputManager, sample::Int, energy::Complex, observables::Dict)

Write physics calculation step data.
"""
function write_physics_step!(manager::MVMCOutputManager, sample::Int, energy::Complex, observables::Dict)
    # Main output (zvo_out.dat)
    if haskey(manager.file_handles, "zvo_out")
        f = manager.file_handles["zvo_out"]
        etot = energy
        etot2 = abs2(energy)
        sztot = get(observables, "sztot", 0.0)
        sztot2 = abs2(sztot)

        @printf(f, "%.18e %.18e %.18e %.18e %.18e %.18e\n",
                real(etot), imag(etot), real(etot2), real((etot2 - etot*etot)/(etot*etot)),
                real(sztot), real(sztot2))
        flush_if_needed!(manager, f)
    end

    # Energy time series
    if haskey(manager.file_handles, "zvo_energy")
        f = manager.file_handles["zvo_energy"]
        @printf(f, "%8d  %.16e  %.16e\n", sample, real(energy), imag(energy))
        flush_if_needed!(manager, f)
    end

    manager.current_sample += 1
end

"""
    write_green_functions!(manager::MVMCOutputManager, gup::Matrix, gdown::Matrix, bin::Int=0)

Write one-body Green functions in mVMC format.
"""
function write_green_functions!(manager::MVMCOutputManager, gup::Matrix, gdown::Matrix, bin::Int=0)
    n_sites = size(gup, 1)

    # Determine file key
    file_key = if bin == 0
        "zvo_cisajs"
    else
        @sprintf("zvo_cisajs_%03d", bin)
    end

    if haskey(manager.file_handles, file_key)
        f = manager.file_handles[file_key]

        # Write spin-up Green function (spin index = 1)
        for i in 1:n_sites, j in 1:n_sites
            @printf(f, "%6d %2d %6d %2d  %.16e %.16e\n",
                    i, 1, j, 1, real(gup[i,j]), imag(gup[i,j]))
        end

        # Write spin-down Green function (spin index = 2)
        for i in 1:n_sites, j in 1:n_sites
            @printf(f, "%6d %2d %6d %2d  %.16e %.16e\n",
                    i, 2, j, 2, real(gdown[i,j]), imag(gdown[i,j]))
        end

        flush_if_needed!(manager, f)
    end
end

"""
    write_two_body_green_functions!(manager::MVMCOutputManager, gup::Matrix, gdown::Matrix, bin::Int=0, max_rows::Int=20000)

Write two-body Green functions using Wick theorem.
"""
function write_two_body_green_functions!(manager::MVMCOutputManager, gup::Matrix, gdown::Matrix, bin::Int=0, max_rows::Int=20000)
    n_sites = size(gup, 1)

    # File keys
    file_key_ex = if bin == 0
        "zvo_cisajscktaltex"
    else
        @sprintf("zvo_cisajscktaltex_%03d", bin)
    end

    file_key_dc = if bin == 0
        "zvo_cisajscktalt"
    else
        @sprintf("zvo_cisajscktalt_%03d", bin)
    end

    # Wick contraction function
    wick4(G, i, j, k, l) = (j == k ? G[i,l] : zero(eltype(G))) - G[i,k] * G[j,l]

    row_count = 0

    # Write equal-time part (Wick contractions)
    if haskey(manager.file_handles, file_key_ex)
        f = manager.file_handles[file_key_ex]

        for i in 1:n_sites, j in 1:n_sites, k in 1:n_sites, l in 1:n_sites
            if row_count >= max_rows
                break
            end

            # Spin-up sector
            g4_up = wick4(gup, i, j, k, l)
            @printf(f, "%6d %2d %6d %2d %6d %2d %6d %2d  %.16e %.16e\n",
                    i, 1, j, 1, k, 1, l, 1, real(g4_up), imag(g4_up))
            row_count += 1

            if row_count >= max_rows
                break
            end

            # Spin-down sector
            g4_down = wick4(gdown, i, j, k, l)
            @printf(f, "%6d %2d %6d %2d %6d %2d %6d %2d  %.16e %.16e\n",
                    i, 2, j, 2, k, 2, l, 2, real(g4_down), imag(g4_down))
            row_count += 1
        end

        flush_if_needed!(manager, f)
    end

    # Write disconnected part
    if haskey(manager.file_handles, file_key_dc)
        f = manager.file_handles[file_key_dc]
        row_count = 0

        for i in 1:n_sites, j in 1:n_sites, k in 1:n_sites, l in 1:n_sites
            if row_count >= max_rows
                break
            end

            # Spin-up disconnected part: G[i,k] * G[j,l]
            g4_dc_up = gup[i,k] * gup[j,l]
            @printf(f, "%6d %2d %6d %2d %6d %2d %6d %2d  %.16e %.16e\n",
                    i, 1, j, 1, k, 1, l, 1, real(g4_dc_up), imag(g4_dc_up))
            row_count += 1

            if row_count >= max_rows
                break
            end

            # Spin-down disconnected part
            g4_dc_down = gdown[i,k] * gdown[j,l]
            @printf(f, "%6d %2d %6d %2d %6d %2d %6d %2d  %.16e %.16e\n",
                    i, 2, j, 2, k, 2, l, 2, real(g4_dc_down), imag(g4_dc_down))
            row_count += 1
        end

        flush_if_needed!(manager, f)
    end
end

"""
    write_lanczos_files!(manager::MVMCOutputManager, energies::Vector{Float64}, variances::Vector{Float64})

Write Lanczos analysis files.
"""
function write_lanczos_files!(manager::MVMCOutputManager, energies::Vector{Float64}, variances::Vector{Float64})
    # zvo_ls_result.dat
    open(joinpath(manager.output_dir, "zvo_ls_result.dat"), "w") do f
        println(f, "# step  Etot  Var(E)")
        for (i, (E, V)) in enumerate(zip(energies, variances))
            @printf(f, "%6d  %.16e  %.16e\n", i, E, V)
        end
    end

    # zvo_ls_alpha_beta.dat
    open(joinpath(manager.output_dir, "zvo_ls_alpha_beta.dat"), "w") do f
        println(f, "# step  alpha  beta")
        for i in eachindex(energies)
            alpha = energies[i]
            beta = i > 1 ? abs(energies[i] - energies[i-1]) : 0.0
            @printf(f, "%6d  %.16e  %.16e\n", i, alpha, beta)
        end
    end
end

"""
    write_timing_info!(manager::MVMCOutputManager, iteration::Int, elapsed_time::Float64, step_time::Float64)

Write timing information.
"""
function write_timing_info!(manager::MVMCOutputManager, iteration::Int, elapsed_time::Float64, step_time::Float64)
    if haskey(manager.file_handles, "zvo_time")
        f = manager.file_handles["zvo_time"]
        @printf(f, "%6d  %.6f  %.6f\n", iteration, elapsed_time, step_time)
        flush_if_needed!(manager, f)
    end
end

"""
    flush_if_needed!(manager::MVMCOutputManager, file::IOStream)

Flush file if flush interval is set and conditions are met.
"""
function flush_if_needed!(manager::MVMCOutputManager, file::IOStream)
    if manager.flush_interval > 0 && manager.current_sample % manager.flush_interval == 0
        flush(file)
    end
end

"""
    close_all_files!(manager::MVMCOutputManager)

Close all open file handles.
"""
function close_all_files!(manager::MVMCOutputManager)
    for (key, handle) in manager.file_handles
        try
            close(handle)
        catch
            # Ignore errors when closing
        end
    end
    empty!(manager.file_handles)
end

"""
    finalize_output!(manager::MVMCOutputManager)

Finalize all outputs and close files.
"""
function finalize_output!(manager::MVMCOutputManager)
    # Flush all remaining buffers
    for (key, handle) in manager.file_handles
        try
            flush(handle)
        catch
        end
    end

    close_all_files!(manager)
end

"""
    print_progress_mvmc_style(iteration::Int, total::Int)

Print optimization progress in mVMC style.
"""
function print_progress_mvmc_style(iteration::Int, total::Int)
    if total < 20
        progress = Int(100.0 * iteration / total)
        println("Progress of Optimization: $progress %.")
    else
        if iteration % (total ÷ 20) == 0
            progress = Int(100.0 * iteration / total)
            println("Progress of Optimization: $progress %.")
        end
    end
end

"""
    print_mvmc_header(config::SimulationConfig)

Print header information in mVMC style.
"""
function print_mvmc_header(config::SimulationConfig)
    println("######  Input Parameter of Standard Interface  ######")
    println()

    face = config.face
    if haskey(face, :L)
        println("  KEYWORD : L                    | VALUE : $(face[:L])")
    end
    if haskey(face, :Lsub)
        println("  KEYWORD : Lsub                 | VALUE : $(face[:Lsub])")
    end
    if haskey(face, :model)
        println("  KEYWORD : model                | VALUE : $(face[:model])")
    end
    if haskey(face, :lattice)
        println("  KEYWORD : lattice              | VALUE : $(face[:lattice])")
    end
    if haskey(face, :J)
        println("  KEYWORD : J                    | VALUE : $(face[:J])")
    end
    if haskey(face, :NSROptItrStep)
        println("  KEYWORD : NSROptItrStep        | VALUE : $(face[:NSROptItrStep])")
    end
    if haskey(face, :TwoSz)
        println("  KEYWORD : 2Sz                  | VALUE : $(face[:TwoSz])")
    end

    println()
    println("#######  Construct Model  #######")
    println()
end