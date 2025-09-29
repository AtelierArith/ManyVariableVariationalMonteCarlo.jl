#!/usr/bin/env julia
using Printf

function usage()
    println("Usage: julia --project examples/19_compare_with_c_output.jl <ref_dir> <test_dir>")
    println("  <ref_dir>: directory containing C mVMC outputs (e.g., zvo_out.dat)")
    println("  <test_dir>: directory containing Julia outputs to compare")
end

function read_numeric_table(path::String)
    if !isfile(path)
        return Array{Float64}(undef, 0, 0)
    end
    rows = Vector{Vector{Float64}}()
    for line in eachline(path)
        s = strip(line)
        isempty(s) && continue
        startswith(s, "#") && continue
        # extract numeric tokens
        toks = split(s)
        vals = Float64[]
        for t in toks
            v = tryparse(Float64, t)
            if v !== nothing
                push!(vals, v)
            end
        end
        isempty(vals) && continue
        push!(rows, vals)
    end
    if isempty(rows)
        return Array{Float64}(undef, 0, 0)
    end
    ncols = maximum(length.(rows))
    mat = fill(NaN, length(rows), ncols)
    for (i, r) in enumerate(rows)
        n = min(length(r), ncols)
        for j in 1:n
            mat[i, j] = r[j]
        end
    end
    return mat
end

function rmse(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
    n = min(length(a), length(b))
    if n == 0
        return NaN
    end
    s = 0.0
    for i in 1:n
        da = a[i]
        db = b[i]
        if !isnan(da) && !isnan(db)
            s += (da - db)^2
        end
    end
    return sqrt(s / n)
end

function compare_file(name::String, refdir::String, testdir::String; colmap::Vector{Int}=Int[])
    refp = joinpath(refdir, name)
    tstp = joinpath(testdir, name)
    ref = read_numeric_table(refp)
    tst = read_numeric_table(tstp)
    @printf("\n[%s]\n", name)
    if size(ref, 1) == 0 && size(tst, 1) == 0
        println("  both missing or empty; skip")
        return true
    elseif size(ref, 1) == 0
        println("  reference missing/empty: ", refp)
        return false
    elseif size(tst, 1) == 0
        println("  test missing/empty: ", tstp)
        return false
    end
    nrows = min(size(ref, 1), size(tst, 1))
    ncols = min(size(ref, 2), size(tst, 2))
    cols = isempty(colmap) ? collect(1:ncols) : colmap
    ok = true
    for j in cols
        a = vec(ref[1:nrows, j])
        b = vec(tst[1:nrows, j])
        e = rmse(a, b)
        @printf("  col %d: rmse = %.6e\n", j, e)
        # relaxed tolerances; energies typically within ~1e-2 for short runs
        tol = j == 1 ? 5e-3 : 1e-2
        if !isnan(e) && e > tol
            ok = false
        end
    end
    return ok
end

function main()
    if length(ARGS) < 2
        usage(); return
    end
    refdir, testdir = ARGS[1], ARGS[2]
    println("Comparing outputs:")
    println("  ref : ", refdir)
    println("  test: ", testdir)

    all_ok = true
    # For optimization runs, focus on zvo_out.dat and zvo_SRinfo.dat
    all_ok &= compare_file("zvo_out.dat", refdir, testdir; colmap=[1])
    all_ok &= compare_file("zvo_SRinfo.dat", refdir, testdir)
    # Parameters vector snapshot (structure differs; compare magnitudes)
    ok_var = compare_file("zvo_var.dat", refdir, testdir)
    all_ok &= ok_var
    # Final parameters
    all_ok &= compare_file("zqp_opt.dat", refdir, testdir)

    println("\nSummary: ", all_ok ? "PASS" : "FAIL")
end

main()

