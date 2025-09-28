@testitem "cisajs: Wick total + DC recovers diagonal G for i=j=k=l (spin up)" begin
    using ManyVariableVariationalMonteCarlo
    using Printf

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
            parts = split(strip(line))
            if length(parts) == 10
                i  = parse(Int, parts[1]); s = parse(Int, parts[2])
                j  = parse(Int, parts[3]); t = parse(Int, parts[4])
                k  = parse(Int, parts[5]); u = parse(Int, parts[6])
                l  = parse(Int, parts[7]); v = parse(Int, parts[8])
                i0,j0,k0,l0,s0,t0,u0,v0 = i,j,k,l,s,t,u,v
                found = true
                break
            end
        end
    end
    @test found

    # Find G(i,l) from one-body file (matching spin)
    Gil = nothing
    open(joinpath(outdir, "zvo_cisajs.dat"), "r") do f
        for line in eachline(f)
            startswith(line, "#") && continue
            parts = split(strip(line))
            if length(parts) == 6
                i  = parse(Int, parts[1]); s = parse(Int, parts[2])
                j  = parse(Int, parts[3]); t = parse(Int, parts[4])
                re = parse(Float64, parts[5])
                if s == s0 && t == t0 && i == i0 && j == l0
                    Gil = re
                    break
                end
            end
        end
    end
    @test Gil !== nothing
    Gil = Float64(Gil)

    # Helper to extract the ex/DC value for the chosen site
    function find_value(path)
        open(path, "r") do f
            for line in eachline(f)
                startswith(line, "#") && continue
                parts = split(strip(line))
                if length(parts) == 10
                    i  = parse(Int, parts[1]); s = parse(Int, parts[2])
                    j  = parse(Int, parts[3]); t = parse(Int, parts[4])
                    k  = parse(Int, parts[5]); u = parse(Int, parts[6])
                    l  = parse(Int, parts[7]); v = parse(Int, parts[8])
                    re = parse(Float64, parts[9])
                    # im = parse(Float64, parts[10])
                    if s==1 && t==1 && u==1 && v==1 && i==i0 && j==j0 && k==k0 && l==l0
                        return re
                    end
                end
            end
        end
        return nothing
    end
    ex_val = find_value(ex_path)
    dc_val = find_value(joinpath(outdir, "zvo_cisajscktalt.dat"))
    @test ex_val !== nothing && dc_val !== nothing

    # Identity: ex + DC = δ_{j,k} * G[i,l]
    δ = (j0 == k0) ? 1.0 : 0.0
    @test isapprox(ex_val + dc_val, δ * Gil; atol=1e-8, rtol=1e-6)
end
