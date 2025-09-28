@testitem "cisajs: write binned variants when NStoreO>0" begin
    using ManyVariableVariationalMonteCarlo

    face = FaceDefinition()
    push_definition!(face, :model, "FermionHubbard")
    push_definition!(face, :lattice, "chain")
    push_definition!(face, :L, 6)
    push_definition!(face, :nelec, 4)
    push_definition!(face, :NVMCCalMode, 1)
    push_definition!(face, :NVMCSample, 20)
    # request one-body local Green for deterministic content
    push_definition!(face, :OneBodyG, true)
    # request two bins of cisajs outputs
    push_definition!(face, :NStoreO, 2)

    config = SimulationConfig(face)
    layout = ParameterLayout(2, 0, 2, 0)

    sim = VMCSimulation(config, layout)
    initialize_simulation!(sim)
    run_physics_calculation!(sim)

    outdir = mktempdir()
    output_results(sim, outdir)

    # Base files still present
    @test isfile(joinpath(outdir, "zvo_cisajs.dat"))
    @test isfile(joinpath(outdir, "zvo_cisajscktaltex.dat"))
    @test isfile(joinpath(outdir, "zvo_cisajscktalt.dat"))

    # Binned variants should be written (_001 and _002)
    @test isfile(joinpath(outdir, "zvo_cisajs_001.dat"))
    @test isfile(joinpath(outdir, "zvo_cisajs_002.dat"))
    @test isfile(joinpath(outdir, "zvo_cisajscktaltex_001.dat"))
    @test isfile(joinpath(outdir, "zvo_cisajscktaltex_002.dat"))
    @test isfile(joinpath(outdir, "zvo_cisajscktalt_001.dat"))
    @test isfile(joinpath(outdir, "zvo_cisajscktalt_002.dat"))

    # Basic content checks: each binned file should have header + at least one data line
    for name in ("zvo_cisajs_001.dat", "zvo_cisajs_002.dat",
                 "zvo_cisajscktaltex_001.dat", "zvo_cisajscktaltex_002.dat",
                 "zvo_cisajscktalt_001.dat", "zvo_cisajscktalt_002.dat")
        lines = readlines(joinpath(outdir, name))
        @test length(lines) >= 2
    end
end
