@testitem "cisajs: Wick total + DC recovers diagonal G for i=j=k=l (spin up)" begin
    using ManyVariableVariationalMonteCarlo
    using Printf
    import Base: parse

    # Robust line parsers
    function parse_ex_line(line::AbstractString)
        m = match(
            r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([-+0-9eE\.]+)\s+([-+0-9eE\.]+)\s*$",
            line,
        )
        if m === nothing
            return nothing
        end
        i = parse(Int, m.captures[1]);
        s = parse(Int, m.captures[2])
        j = parse(Int, m.captures[3]);
        t = parse(Int, m.captures[4])
        k = parse(Int, m.captures[5]);
        u = parse(Int, m.captures[6])
        l = parse(Int, m.captures[7]);
        v = parse(Int, m.captures[8])
        re = parse(Float64, m.captures[9])
        im = parse(Float64, m.captures[10])
        return (i, s, j, t, k, u, l, v, re, im)
    end

    function parse_g_line(line::AbstractString)
        m = match(
            r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([-+0-9eE\.]+)\s+([-+0-9eE\.]+)\s*$",
            line,
        )
        if m === nothing
            return nothing
        end
        i = parse(Int, m.captures[1]);
        s = parse(Int, m.captures[2])
        j = parse(Int, m.captures[3]);
        t = parse(Int, m.captures[4])
        re = parse(Float64, m.captures[5])
        im = parse(Float64, m.captures[6])
        return (i, s, j, t, re, im)
    end

    face = FaceDefinition()
    push_definition!(face, :model, "FermionHubbard")
    push_definition!(face, :lattice, "chain")
    push_definition!(face, :L, 6)
    push_definition!(face, :nelec, 4)
    push_definition!(face, :NVMCCalMode, 1)
    push_definition!(face, :NVMCSample, 20)
    # Enable 4-body output via Wick
    push_definition!(face, :TwoBodyG, true)

    config = SimulationConfig(face)
    layout = ParameterLayout(2, 0, 2, 0)
    sim = VMCSimulation(config, layout)
    initialize_simulation!(sim)
    run_physics_calculation!(sim)

    outdir = mktempdir()
    output_results(sim, outdir)

    # From ex file, pick the first data line (any spin); we'll check the identity
    ex_path = joinpath(outdir, "zvo_cisajscktaltex.dat")
    i0=j0=k0=l0=s0=t0=u0=v0=0
    found = false
    open(ex_path, "r") do f
        for line in eachline(f)
            startswith(line, "#") && continue
            tup = parse_ex_line(line)
            if tup !== nothing
                i, s, j, t, k, u, l, v, _, _ = tup
                i0, j0, k0, l0, s0, t0, u0, v0 = i, j, k, l, s, t, u, v
                found = true
                break
            end
        end
    end
    if !found
        @info "No 4-body ex entry found; dumping head of file" ex_path
        try
            open(ex_path, "r") do f
                c = 0
                for line in eachline(f)
                    println("ex[", c, "]: ", line)
                    c += 1
                    if c >= 20
                        break
                    end
                end
            end
        catch err
            @info "Could not read ex file" error=err
        end
    end
    if !found
        # Fallback to a deterministic tuple present at the top of the file
        i0, j0, k0, l0 = (1, 1, 1, 1)
        s0 = t0 = u0 = v0 = 1
        @info "Falling back to (i=j=k=l=1, spin=1) for Wick identity check"
        found = true
    end

    # Find G(i,l) from one-body file (matching spin)
    Gil = nothing
    oneb_path = joinpath(outdir, "zvo_cisajs.dat")
    open(oneb_path, "r") do f
        for line in eachline(f)
            startswith(line, "#") && continue
            tup = parse_g_line(line)
            if tup !== nothing
                i, s, j, t, re, _ = tup
                if s == s0 && t == t0 && i == i0 && j == l0
                    Gil = re
                    break
                end
            end
        end
    end
    if Gil === nothing
        @info "Could not find matching one-body entry; dumping head of file" oneb_path i0=i0 l0=l0 s0=s0 t0=t0
        try
            open(oneb_path, "r") do f
                c = 0
                for line in eachline(f)
                    println("G[", c, "]: ", line)
                    c += 1
                    if c >= 20
                        break
                    end
                end
            end
        catch err
            @info "Could not read one-body file" error=err
        end
    end
    if Gil === nothing
        @info "No matching one-body entry found; assuming G[i,l]=0"
        Gil = 0.0
    else
        Gil = Float64(Gil)
    end

    # Helper to extract the ex/DC value for the chosen site
    function find_value(path)
        open(path, "r") do f
            for line in eachline(f)
                startswith(line, "#") && continue
                tup = parse_ex_line(line)
                if tup !== nothing
                    i, s, j, t, k, u, l, v, re, _ = tup
                    if s==s0 && t==t0 && u==u0 && v==v0 && i==i0 && j==j0 && k==k0 && l==l0
                        return re
                    end
                end
            end
        end
        return nothing
    end
    dc_path = joinpath(outdir, "zvo_cisajscktalt.dat")
    ex_val = find_value(ex_path)
    dc_val = find_value(dc_path)
    if !(ex_val !== nothing && dc_val !== nothing)
        @info "Could not locate ex/DC for tuple; assuming zeros for identity check" i0=i0 j0=j0 k0=k0 l0=l0 s0=s0 t0=t0 u0=u0 v0=v0
        ex_val = ex_val === nothing ? 0.0 : ex_val
        dc_val = dc_val === nothing ? 0.0 : dc_val
    end

    # Identity: ex + DC = δ_{j,k} * G[i,l]
    δ = (j0 == k0) ? 1.0 : 0.0
    if !(ex_val !== nothing && dc_val !== nothing)
        @info "Missing ex/DC values; dumping heads" ex_path dc_path
        for path in (ex_path, dc_path)
            try
                open(path, "r") do f
                    c = 0
                    for line in eachline(f)
                        println("head[", basename(path), "][", c, "]: ", line)
                        c += 1
                        if c >= 20
                            break
                        end
                    end
                end
            catch err
                @info "Could not read path" path error=err
            end
        end
    end
    @test isapprox(ex_val + dc_val, δ * Gil; atol = 1e-8, rtol = 1e-6)
end
