#!/usr/bin/env julia

using Printf

function read_energy_series(path::String)
    if !isfile(path)
        return Float64[]
    end
    energies = Float64[]
    open(path, "r") do f
        for line in eachline(f)
            s = strip(line)
            isempty(s) && continue
            startswith(s, "#") && continue
            parts = split(s)
            # First column is Re[E]
            try
                push!(energies, parse(Float64, parts[1]))
            catch
                # Skip malformed lines
            end
        end
    end
    return energies
end

function compare_series(julia_path::String, c_path::String; rtol=1e-3, atol=1e-5)
    ej = read_energy_series(julia_path)
    ec = read_energy_series(c_path)

    if isempty(ej)
        return (false, "Julia energy series empty at " * julia_path)
    end
    if isempty(ec)
        return (false, "C energy series empty at " * c_path)
    end

    n = min(length(ej), length(ec))
    ejv = ej[1:n]
    ecv = ec[1:n]
    diffs = abs.(ejv .- ecv)

    maxdiff = maximum(diffs)
    reldiffs = diffs ./ max.(abs.(ecv), eps())
    maxrdiff = maximum(reldiffs)

    pass = (maxdiff ≤ atol) || (maxrdiff ≤ rtol)
    return (pass, @sprintf("n=%d  max|Δ|=%.3e  max(relΔ)=%.3e  ej_end=%.6f  ec_end=%.6f",
                           n, maxdiff, maxrdiff, ejv[end], ecv[end]))
end

function compare_final_only(julia_path::String, c_path::String; rtol=1e-3, atol=1e-5)
    ej = read_energy_series(julia_path)
    ec = read_energy_series(c_path)
    if isempty(ej) || isempty(ec)
        return (false, "empty energy series")
    end
    Ej = ej[end]
    Ec = ec[end]
    Δ = abs(Ej - Ec)
    relΔ = Δ / max(abs(Ec), eps())
    pass = (Δ ≤ atol) || (relΔ ≤ rtol)
    return (pass, @sprintf("|Δ|=%.3e relΔ=%.3e (Ej=%.6f, Ec=%.6f)", Δ, relΔ, Ej, Ec))
end

function parse_args(args)
    opts = Dict{String,Any}(
        "root" => length(args) ≥ 1 && !startswith(args[1], "--") ? args[1] : joinpath(@__DIR__, "..", "mVMC", "samples", "Standard", "Spin", "HeisenbergChain"),
        "rtol" => 1e-3,
        "atol" => 1e-5,
        "final_only" => false,
    )

    for a in args
        if a == "--final-only"
            opts["final_only"] = true
        elseif startswith(a, "--rtol=")
            opts["rtol"] = parse(Float64, split(a, "=", limit=2)[2])
        elseif startswith(a, "--atol=")
            opts["atol"] = parse(Float64, split(a, "=", limit=2)[2])
        end
    end
    return opts
end

function main()
    opts = parse_args(ARGS)
    root = opts["root"]
    stdface = joinpath(root, "StdFace.def")
    if !isfile(stdface)
        println("[error] StdFace.def not found at: ", stdface)
        println("Usage: julia --project examples/19_compare_with_c_spin_chain.jl <path/to/HeisenbergChain>")
        return
    end

    # Paths
    julia_out = joinpath(root, "zvo_out.dat")
    c_out_dir = joinpath(root, "output")
    c_out_001 = joinpath(c_out_dir, "zvo_out_001.dat")
    c_out = isfile(c_out_001) ? c_out_001 : joinpath(c_out_dir, "zvo_out.dat")

    rtol = opts["rtol"]; atol = opts["atol"]

    println("Comparing energy series:")
    println("  Julia: ", julia_out)
    println("  C    : ", c_out)

    if !opts["final_only"]
        pass, summary = compare_series(julia_out, c_out; rtol=rtol, atol=atol)
        println("  Result(series): ", pass ? "PASS" : "FAIL")
        println("  Stats(series) : ", summary)
    end

    # Also compare final energies only (looser criterion)
    passF, summaryF = compare_final_only(julia_out, c_out; rtol=rtol, atol=atol)
    println("  Result(final) : ", passF ? "PASS" : "FAIL")
    println("  Stats(final)  : ", summaryF)

    # Exit code-like message for CI visibility
    if (opts["final_only"] && !passF) || (!opts["final_only"] && (!passF))
        println("Note: Energy series do not match within tolerance.")
        println("      Check simulation mode/parameters and RNG seeding.")
    end
end

main()
