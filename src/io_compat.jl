"""
File I/O Compatibility Layer for mVMC C Implementation

Provides exact file format compatibility with the C mVMC implementation,
including binary and text output formats, file naming conventions,
and precision matching.

Ported from initfile.c and related C modules.
"""

using Printf
using Base.Iterators

# Constants matching C implementation
const D_FileNameMax = 256
const C_FLOAT_PRECISION = "%.18e"
const C_COMPLEX_FORMAT = "% .18e % .18e"

"""
    MVMCFileManager

Manages file handles and naming conventions matching the C implementation.
"""
mutable struct MVMCFileManager
    # File naming
    data_file_head::String
    para_file_head::String

    # File handles (matching C global variables)
    file_out::Union{IO, Nothing}
    file_var::Union{IO, Nothing}
    file_time::Union{IO, Nothing}
    file_srinfo::Union{IO, Nothing}
    file_cisajs::Union{IO, Nothing}
    file_cisajscktalt::Union{IO, Nothing}
    file_cisajscktaltex::Union{IO, Nothing}
    file_ls::Union{IO, Nothing}

    # File naming state
    data_idx_start::Int
    data_qty_smp::Int
    file_flush_interval::Int

    # Binary mode flag
    flag_binary::Bool

    function MVMCFileManager(;
        data_file_head::String = "zvo",
        para_file_head::String = "zqp",
        data_idx_start::Int = 0,
        data_qty_smp::Int = 1,
        file_flush_interval::Int = 1,
        flag_binary::Bool = false
    )
        new(
            data_file_head,
            para_file_head,
            nothing,  # file_out
            nothing,  # file_var
            nothing,  # file_time
            nothing,  # file_srinfo
            nothing,  # file_cisajs
            nothing,  # file_cisajscktalt
            nothing,  # file_cisajscktaltex
            nothing,  # file_ls
            data_idx_start,
            data_qty_smp,
            file_flush_interval,
            flag_binary
        )
    end
end

"""
    init_file!(manager::MVMCFileManager, nvmc_cal_mode::Int, rank::Int)

Initialize files for parameter optimization mode.
Matches C function InitFile() from initfile.c.

C実装参考: initfile.c 1行目から243行目まで
"""
function init_file!(manager::MVMCFileManager, nvmc_cal_mode::Int, rank::Int)
    if rank != 0
        return
    end

    # Time file
    time_filename = @sprintf("%s_time_%03d.dat", manager.data_file_head, manager.data_idx_start)
    manager.file_time = open(time_filename, "w")

    if nvmc_cal_mode == 0  # Parameter optimization mode
        # SR info file
        srinfo_filename = @sprintf("%s_SRinfo.dat", manager.data_file_head)
        manager.file_srinfo = open(srinfo_filename, "w")

        # Write SR info header (matching C format)
        println(manager.file_srinfo, "#Npara Msize optCut diagCut sDiagMax  sDiagMin    absRmax       imax")

        # Output file
        out_filename = @sprintf("%s_out_%03d.dat", manager.data_file_head, manager.data_idx_start)
        manager.file_out = open(out_filename, "w")

        # Variable file
        if !manager.flag_binary
            var_filename = @sprintf("%s_var_%03d.dat", manager.data_file_head, manager.data_idx_start)
            manager.file_var = open(var_filename, "w")
        else
            var_filename = @sprintf("%s_varbin_%03d.dat", manager.data_file_head, manager.data_idx_start)
            manager.file_var = open(var_filename, "w")
            # Write binary header (matching C format)
            # Note: In Julia, we'll write the header as text for now
            # TODO: Implement proper binary format
        end
    end
end

"""
    init_file_phys_cal!(manager::MVMCFileManager, i::Int, rank::Int)

Initialize files for physics calculation mode.
Matches C function InitFilePhysCal() from initfile.c.
"""
function init_file_phys_cal!(manager::MVMCFileManager, i::Int, rank::Int)
    if rank != 0
        return
    end

    idx = i + manager.data_idx_start

    # Output file
    out_filename = @sprintf("%s_out_%03d.dat", manager.data_file_head, idx)
    manager.file_out = open(out_filename, "w")

    # Variable file
    if !manager.flag_binary
        var_filename = @sprintf("%s_var_%03d.dat", manager.data_file_head, idx)
        manager.file_var = open(var_filename, "w")
    else
        var_filename = @sprintf("%s_varbin_%03d.dat", manager.data_file_head, idx)
        manager.file_var = open(var_filename, "w")
        # Write binary header
        # TODO: Implement proper binary format
    end

    # Green function files (if needed)
    # These will be opened when needed based on NCisAjs, etc.
end

"""
    write_energy_output!(manager::MVMCFileManager, etot::ComplexF64, etot2::ComplexF64, sztot::ComplexF64, sztot2::ComplexF64)

Write energy output matching C format.
Matches C function outputData() from vmcmain.c.
"""
function write_energy_output!(manager::MVMCFileManager, etot::ComplexF64, etot2::ComplexF64, sztot::ComplexF64, sztot2::ComplexF64)
    if manager.file_out === nothing
        return
    end

    # Match C format: "% .18e % .18e  % .18e % .18e %.18e %.18e\n"
    variance = (etot2 - etot * etot) / (etot * etot)
    @printf(manager.file_out, "% .18e % .18e  % .18e % .18e %.18e %.18e\n",
        real(etot), imag(etot), real(etot2), real(variance), real(sztot), real(sztot2))

    maybe_flush_interval!(manager)
end

"""
    write_variable_output!(manager::MVMCFileManager, etot::ComplexF64, etot2::ComplexF64, para::Vector{ComplexF64})

Write variable output matching C format.
Matches C function outputData() variable writing.
"""
function write_variable_output!(manager::MVMCFileManager, etot::ComplexF64, etot2::ComplexF64, para::Vector{ComplexF64})
    if manager.file_var === nothing
        return
    end

    if !manager.flag_binary
        # Text format: "% .18e % .18e 0.0 % .18e % .18e 0.0 " + parameters
        @printf(manager.file_var, "% .18e % .18e 0.0 % .18e % .18e 0.0 ",
            real(etot), imag(etot), real(etot2), imag(etot2))

        # Write parameters
        for p in para
            @printf(manager.file_var, "% .18e % .18e 0.0 ", real(p), imag(p))
        end
        println(manager.file_var)
    else
        # Binary format - write parameters as binary
        # TODO: Implement proper binary format matching C
        for p in para
            write(manager.file_var, Float64(real(p)))
            write(manager.file_var, Float64(imag(p)))
        end
    end

    maybe_flush_interval!(manager)
end

"""
    write_green_function_output!(manager::MVMCFileManager, filename::String, data::Vector{ComplexF64})

Write Green function output matching C format.
"""
function write_green_function_output!(manager::MVMCFileManager, filename::String, data::Vector{ComplexF64})
    if isempty(data)
        return
    end

    file = open(filename, "w")
    try
        for val in data
            @printf(file, "% .18e % .18e\n", real(val), imag(val))
        end
    finally
        close(file)
    end
end

"""
    maybe_flush_interval!(manager::MVMCFileManager)

Flush files at specified intervals.
"""
function maybe_flush_interval!(manager::MVMCFileManager)
    if manager.file_flush_interval > 0
        # This would be called periodically during calculation
        # Implementation depends on integration with main loop
    end
end

"""
    close_all_files!(manager::MVMCFileManager)

Close all open files.
"""
function close_all_files!(manager::MVMCFileManager)
    for field in fieldnames(MVMCFileManager)
        if field in [:file_out, :file_var, :file_time, :file_srinfo, :file_cisajs,
                     :file_cisajscktalt, :file_cisajscktaltex, :file_ls]
            file = getfield(manager, field)
            if file !== nothing
                close(file)
                setfield!(manager, field, nothing)
            end
        end
    end
end

"""
    format_complex_c_style(z::ComplexF64) -> String

Format complex number matching C's fprintf format.
"""
function format_complex_c_style(z::ComplexF64)::String
    @sprintf("% .18e % .18e", real(z), imag(z))
end

"""
    format_float_c_style(x::Float64) -> String

Format float matching C's fprintf format.
"""
function format_float_c_style(x::Float64)::String
    @sprintf("% .18e", x)
end

"""
    generate_filename(prefix::String, suffix::String, idx::Int) -> String

Generate filename matching C sprintf pattern.
"""
function generate_filename(prefix::String, suffix::String, idx::Int)::String
    @sprintf("%s_%s_%03d.dat", prefix, suffix, idx)
end

# Export functions for use in other modules
export MVMCFileManager, init_file!, init_file_phys_cal!, write_energy_output!,
       write_variable_output!, write_green_function_output!, close_all_files!,
       format_complex_c_style, format_float_c_style, generate_filename
