"""
mVMC Test Validation Framework

Statistical validation framework for comparing Julia implementation results
with C reference implementation results.

Based on mVMC/test/python/runtest.py validation logic:
- Compare calculated values with reference mean
- Check if difference is within 3 standard deviations
- Apply minimum absolute error threshold (1e-8)
"""

using Printf
using Statistics

"""
    ValidationResult

Result of validation comparison between calculated and reference values.
"""
struct ValidationResult
    parameter_name::String
    calculated_value::Float64
    reference_mean::Float64
    reference_std::Float64
    difference::Float64
    within_3sigma::Bool
    passes_threshold::Bool
    passes::Bool

    function ValidationResult(
        name::String,
        calc::Float64,
        ref_mean::Float64,
        ref_std::Float64,
    )
        diff = abs(calc - ref_mean)
        within_3sigma = diff < 3 * ref_std
        passes_threshold = !(diff >= 3 * ref_std && diff >= 1e-8)
        passes = passes_threshold

        new(name, calc, ref_mean, ref_std, diff, within_3sigma, passes_threshold, passes)
    end
end

"""
    read_mvmc_output(filename::String) -> Vector{Float64}

Read mVMC output file (zqp_opt.dat format).
Returns array of float values (first two columns: real and imaginary parts).
"""
function read_mvmc_output(filename::String)
    values = Float64[]

    if !isfile(filename)
        error("Output file not found: $filename")
    end

    open(filename, "r") do f
        for line in eachline(f)
            line = strip(line)
            if isempty(line) || startswith(line, "#")
                continue
            end

            parts = split(line)
            if length(parts) >= 2
                # Read real part
                push!(values, parse(Float64, parts[1]))
                # Read imaginary part
                push!(values, parse(Float64, parts[2]))
            end
        end
    end

    return values
end

"""
    read_reference_data(ref_mean_file::String, ref_std_file::String) -> (Vector{Float64}, Vector{Float64})

Read reference mean and standard deviation data.
"""
function read_reference_data(ref_mean_file::String, ref_std_file::String)
    ref_mean = Float64[]
    ref_std = Float64[]

    if !isfile(ref_mean_file)
        error("Reference mean file not found: $ref_mean_file")
    end
    if !isfile(ref_std_file)
        error("Reference std file not found: $ref_std_file")
    end

    # Read mean values
    open(ref_mean_file, "r") do f
        for line in eachline(f)
            line = strip(line)
            if !isempty(line) && !startswith(line, "#")
                push!(ref_mean, parse(Float64, line))
            end
        end
    end

    # Read std values
    open(ref_std_file, "r") do f
        for line in eachline(f)
            line = strip(line)
            if !isempty(line) && !startswith(line, "#")
                push!(ref_std, parse(Float64, line))
            end
        end
    end

    if length(ref_mean) != length(ref_std)
        error("Reference mean and std have different lengths")
    end

    return (ref_mean, ref_std)
end

"""
    validate_against_reference(
        calc_values::Vector{Float64},
        ref_mean::Vector{Float64},
        ref_std::Vector{Float64};
        n_compare::Int = 2
    ) -> (Bool, Vector{ValidationResult})

Validate calculated values against reference data.

# Arguments
- `calc_values`: Calculated values from Julia implementation
- `ref_mean`: Reference mean values from C implementation
- `ref_std`: Reference standard deviation values
- `n_compare`: Number of values to compare (default: 2, first two parameters)

# Returns
- `passed`: Overall pass/fail status
- `results`: Detailed validation results for each parameter

# Validation Criteria (from mVMC test suite)
For each parameter:
- Pass if: |calc - ref_mean| < 3 * ref_std OR |calc - ref_mean| < 1e-8
- Fail otherwise
"""
function validate_against_reference(
    calc_values::Vector{Float64},
    ref_mean::Vector{Float64},
    ref_std::Vector{Float64};
    n_compare::Int = 2,
)
    if length(calc_values) < n_compare
        error("Not enough calculated values: got $(length(calc_values)), need $n_compare")
    end
    if length(ref_mean) < n_compare || length(ref_std) < n_compare
        error("Not enough reference values")
    end

    results = ValidationResult[]
    overall_pass = true

    for i in 1:n_compare
        param_name = i == 1 ? "Parameter 1 (real)" : "Parameter 2 (imag)"
        result = ValidationResult(
            param_name,
            calc_values[i],
            ref_mean[i],
            ref_std[i],
        )

        push!(results, result)

        if !result.passes
            overall_pass = false
        end
    end

    return (overall_pass, results)
end

"""
    print_validation_report(results::Vector{ValidationResult})

Print a human-readable validation report.
"""
function print_validation_report(results::Vector{ValidationResult})
    println()
    println("=" ^ 80)
    println("Validation Report")
    println("=" ^ 80)
    println()

    for result in results
        println("Parameter: $(result.parameter_name)")
        println("  Calculated:     $(result.calculated_value)")
        println("  Reference mean: $(result.reference_mean)")
        println("  Reference std:  $(result.reference_std)")
        println("  Difference:     $(result.difference)")
        println("  Within 3σ:      $(result.within_3sigma)")
        println("  Status:         $(result.passes ? "✓ PASS" : "✗ FAIL")")
        println()
    end

    n_passed = count(r -> r.passes, results)
    n_total = length(results)

    println("=" ^ 80)
    println("Summary: $n_passed / $n_total tests passed")
    println("=" ^ 80)
    println()
end

"""
    run_mvmc_validation_test(
        output_file::String,
        ref_mean_file::String,
        ref_std_file::String;
        n_compare::Int = 2,
        verbose::Bool = true
    ) -> Bool

Run full validation test comparing Julia output with C reference.

# Arguments
- `output_file`: Path to Julia output file (e.g., "output/zqp_opt.dat")
- `ref_mean_file`: Path to reference mean file (e.g., "ref/ref_mean.dat")
- `ref_std_file`: Path to reference std file (e.g., "ref/ref_std.dat")
- `n_compare`: Number of parameters to compare (default: 2)
- `verbose`: Print detailed report (default: true)

# Returns
- `true` if all validation tests pass
- `false` if any test fails
"""
function run_mvmc_validation_test(
    output_file::String,
    ref_mean_file::String,
    ref_std_file::String;
    n_compare::Int = 2,
    verbose::Bool = true,
)
    # Read data
    calc_values = read_mvmc_output(output_file)
    ref_mean, ref_std = read_reference_data(ref_mean_file, ref_std_file)

    # Validate
    passed, results = validate_against_reference(
        calc_values,
        ref_mean,
        ref_std;
        n_compare = n_compare,
    )

    # Print report
    if verbose
        print_validation_report(results)
    end

    return passed
end

"""
    compare_output_files(julia_file::String, c_file::String; rtol::Float64=1e-6)

Direct comparison of output files (element by element).

Returns true if files match within relative tolerance.
"""
function compare_output_files(julia_file::String, c_file::String; rtol::Float64 = 1e-6)
    julia_data = read_mvmc_output(julia_file)
    c_data = read_mvmc_output(c_file)

    if length(julia_data) != length(c_data)
        @warn "File lengths differ: Julia=$(length(julia_data)), C=$(length(c_data))"
        return false
    end

    max_diff = 0.0
    mismatches = 0

    for (i, (j_val, c_val)) in enumerate(zip(julia_data, c_data))
        diff = abs(j_val - c_val)
        rel_diff = diff / (abs(c_val) + 1e-15)  # Avoid division by zero

        if rel_diff > rtol
            mismatches += 1
            if diff > max_diff
                max_diff = diff
            end
        end
    end

    if mismatches > 0
        @warn "Found $mismatches mismatches (max diff: $max_diff)"
        return false
    end

    return true
end
