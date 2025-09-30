"""
zqp_opt.dat Writer for mVMC Compatibility

Writes optimized variational parameters in the exact format expected by mVMC.
Format: "% .18e % .18e 0.0 " (real part, imaginary part, always 0.0 for third column)
"""

using Printf

"""
    write_zqp_opt_file(filename::String, parameters::Vector{T}) where T

Write optimized parameters to zqp_opt.dat file in mVMC-compatible format.

# Arguments
- `filename`: Output file path (usually "zqp_opt.dat")
- `parameters`: Vector of variational parameters (ComplexF64 or Float64)

# Format
Each parameter is written as:
    real_part imaginary_part 0.0
All with "% .18e" precision (18 decimal places in scientific notation)
"""
function write_zqp_opt_file(filename::String, parameters::Vector{T}) where {T}
    open(filename, "w") do f
        for param in parameters
            if T <: Complex
                @printf(f, "% .18e % .18e 0.0\n", real(param), imag(param))
            else
                # Real parameters: imaginary part is 0.0
                @printf(f, "% .18e % .18e 0.0\n", param, 0.0)
            end
        end
    end
    return nothing
end

"""
    write_zqp_opt_file_split(output_dir::String, params_all::Vector{T},
                             n_proj::Int, n_slater::Int, n_opttrans::Int) where T

Write zqp_opt.dat and split parameter files in mVMC format.

Writes:
- zqp_opt.dat: All parameters
- zqp_gutzwiller_opt.dat: Projection parameters (Gutzwiller + Jastrow)
- zqp_orbital_opt.dat: Slater (orbital) parameters
- zqp_opttrans_opt.dat: OptTrans parameters (if any)

# Arguments
- `output_dir`: Output directory path
- `params_all`: All variational parameters
- `n_proj`: Number of projection parameters
- `n_slater`: Number of Slater parameters
- `n_opttrans`: Number of OptTrans parameters
"""
function write_zqp_opt_file_split(
    output_dir::String,
    params_all::Vector{T},
    n_proj::Int,
    n_slater::Int,
    n_opttrans::Int,
) where {T}
    # Write all parameters
    write_zqp_opt_file(joinpath(output_dir, "zqp_opt.dat"), params_all)

    # Split parameters
    idx = 1

    # Projection parameters (Gutzwiller + Jastrow)
    if n_proj > 0
        proj_params = params_all[idx:(idx+n_proj-1)]
        write_zqp_opt_file(joinpath(output_dir, "zqp_gutzwiller_opt.dat"), proj_params)
        idx += n_proj
    end

    # Slater (Orbital) parameters
    if n_slater > 0
        slater_params = params_all[idx:(idx+n_slater-1)]
        write_zqp_opt_file(joinpath(output_dir, "zqp_orbital_opt.dat"), slater_params)
        idx += n_slater
    end

    # OptTrans parameters
    if n_opttrans > 0
        opttrans_params = params_all[idx:(idx+n_opttrans-1)]
        write_zqp_opt_file(joinpath(output_dir, "zqp_opttrans_opt.dat"), opttrans_params)
    end

    return nothing
end

"""
    read_zqp_opt_file(filename::String) -> Vector{ComplexF64}

Read optimized parameters from zqp_opt.dat file in mVMC format.

Returns a vector of complex parameters (even if imaginary parts are zero).
"""
function read_zqp_opt_file(filename::String)
    parameters = ComplexF64[]

    open(filename, "r") do f
        for line in eachline(f)
            # Skip empty lines and comments
            line = strip(line)
            if isempty(line) || startswith(line, "#")
                continue
            end

            # Parse line: real imag 0.0
            parts = split(line)
            if length(parts) >= 2
                re = parse(Float64, parts[1])
                im = parse(Float64, parts[2])
                push!(parameters, complex(re, im))
            end
        end
    end

    return parameters
end

"""
    validate_zqp_opt_format(filename::String) -> Bool

Validate that a zqp_opt.dat file has the correct format.

Returns true if valid, false otherwise.
Prints warnings for format issues.
"""
function validate_zqp_opt_format(filename::String)
    if !isfile(filename)
        @warn "File not found: $filename"
        return false
    end

    valid = true
    line_num = 0

    open(filename, "r") do f
        for line in eachline(f)
            line_num += 1
            line = strip(line)

            # Skip empty lines
            if isempty(line)
                continue
            end

            # Parse line
            parts = split(line)
            if length(parts) != 3
                @warn "Line $line_num: Expected 3 values, got $(length(parts))"
                valid = false
                continue
            end

            # Check if third column is 0.0
            try
                third_val = parse(Float64, parts[3])
                if third_val != 0.0
                    @warn "Line $line_num: Third column should be 0.0, got $third_val"
                end
            catch e
                @warn "Line $line_num: Cannot parse third column: $(parts[3])"
                valid = false
            end
        end
    end

    return valid
end
