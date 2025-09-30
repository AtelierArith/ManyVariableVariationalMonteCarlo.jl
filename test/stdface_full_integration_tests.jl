# StdFace Full Integration Tests
# End-to-end tests for the complete run_mvmc_from_stdface workflow

@testitem "run_mvmc_from_stdface basic workflow with fixture" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    fixture_path = joinpath(@__DIR__, "fixtures", "simple_spin_chain.def")

    if isfile(fixture_path)
        output_dir = mktempdir()

        try
            # Run the complete workflow
            sim = run_mvmc_from_stdface(fixture_path; T=ComplexF64, output_dir=output_dir)

            # Test that simulation object is created
            @test sim isa ManyVariableVariationalMonteCarlo.EnhancedVMCSimulation{ComplexF64}
            @test sim.config.nsites == 4
            @test sim.config.model == :Spin
            @test sim.config.lattice == :chain

            # Test that VMC state is initialized
            @test sim.vmc_state !== nothing
            @test sim.vmc_state.n_sites == 4
            @test sim.vmc_state.n_electrons == 4

            # Test that quantum projection is initialized
            @test sim.quantum_projection !== nothing
            @test sim.quantum_projection.is_initialized == true

            # Test that expert mode files are generated
            @test isfile(joinpath(output_dir, "locspn.def"))
            @test isfile(joinpath(output_dir, "trans.def"))
            @test isfile(joinpath(output_dir, "modpara.def"))
            @test isfile(joinpath(output_dir, "namelist.def"))

            # Test that simulation mode is correctly determined
            @test sim.mode == PHYSICS_CALCULATION  # NVMCCalMode = 1 in fixture

        finally
            rm(output_dir, recursive=true, force=true)
        end
    else
        @test_skip "Test fixture file not available"
    end
end

@testitem "run_mvmc_from_stdface with parameter optimization mode" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    # Create a temporary StdFace.def for optimization mode
    test_content = """
    L = 4
    model = "Spin"
    lattice = "chain"
    J = 1.0
    2Sz = 0

    # Parameter optimization mode
    NVMCCalMode = 0
    NSROptItrStep = 5
    NSROptItrSmp = 3
    NVMCWarmUp = 3
    NVMCSample = 10
    RndSeed = 12345
    """

    temp_file = tempname() * "_opt.def"
    write(temp_file, test_content)
    output_dir = mktempdir()

    try
        sim = run_mvmc_from_stdface(temp_file; T=ComplexF64, output_dir=output_dir)

        @test sim.mode == PARAMETER_OPTIMIZATION
        @test sim.config.nsites == 4

        # Should have optimization results (even if minimal)
        @test !isnothing(sim.optimization_results) || !isnothing(sim.physics_results)

    finally
        rm(temp_file, force=true)
        rm(output_dir, recursive=true, force=true)
    end
end

@testitem "run_mvmc_from_stdface with Hubbard model" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    fixture_path = joinpath(@__DIR__, "fixtures", "simple_hubbard_chain.def")

    if isfile(fixture_path)
        output_dir = mktempdir()

        try
            sim = run_mvmc_from_stdface(fixture_path; T=ComplexF64, output_dir=output_dir)

            @test sim.config.model == :Hubbard
            @test sim.config.lattice == :chain
            @test sim.mode == PARAMETER_OPTIMIZATION  # NVMCCalMode = 0 in fixture

            # Hubbard model should have different parameter layout
            @test length(sim.parameters.proj) > 0  # Gutzwiller parameters

            # Should generate appropriate expert mode files
            @test isfile(joinpath(output_dir, "trans.def"))  # Transfer terms
            @test isfile(joinpath(output_dir, "coulombintra.def"))  # U terms

        finally
            rm(output_dir, recursive=true, force=true)
        end
    else
        @test_skip "Hubbard test fixture file not available"
    end
end

@testitem "run_mvmc_from_stdface with 2D lattice" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    fixture_path = joinpath(@__DIR__, "fixtures", "square_lattice.def")

    if isfile(fixture_path)
        output_dir = mktempdir()

        try
            sim = run_mvmc_from_stdface(fixture_path; T=ComplexF64, output_dir=output_dir)

            @test sim.config.lattice == :square
            @test sim.config.nsites == 4  # 2x2 lattice

            # 2D lattice should work with the same workflow
            @test sim.vmc_state !== nothing
            @test sim.quantum_projection !== nothing

        finally
            rm(output_dir, recursive=true, force=true)
        end
    else
        @test_skip "Square lattice test fixture file not available"
    end
end

@testitem "run_mvmc_from_stdface output file generation" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    fixture_path = joinpath(@__DIR__, "fixtures", "simple_spin_chain.def")

    if isfile(fixture_path)
        output_dir = mktempdir()

        try
            sim = run_mvmc_from_stdface(fixture_path; T=ComplexF64, output_dir=output_dir)

            # Check expert mode files
            expert_files = ["locspn.def", "trans.def", "modpara.def", "namelist.def",
                          "gutzwilleridx.def", "jastrowidx.def", "greenone.def"]

            for file in expert_files
                filepath = joinpath(output_dir, file)
                @test isfile(filepath)
                @test filesize(filepath) > 0
            end

            # For Spin model, should have hund.def
            @test isfile(joinpath(output_dir, "hund.def"))

            # Check that output directory is set correctly
            @test sim.output_manager.output_dir == output_dir

        finally
            rm(output_dir, recursive=true, force=true)
        end
    else
        @test_skip "Test fixture file not available"
    end
end

@testitem "run_mvmc_from_stdface error handling" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    # Test with non-existent file
    @test_throws Exception run_mvmc_from_stdface("nonexistent.def")

    # Test with invalid StdFace.def content
    invalid_content = """
    L = not_a_number
    model = "InvalidModel"
    """

    temp_file = tempname() * "_invalid.def"
    write(temp_file, invalid_content)

    try
        # Should handle parsing errors gracefully
        @test_throws Exception run_mvmc_from_stdface(temp_file)
    finally
        rm(temp_file, force=true)
    end
end

@testitem "run_mvmc_from_stdface with different data types" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    fixture_path = joinpath(@__DIR__, "fixtures", "simple_spin_chain.def")

    if isfile(fixture_path)
        # Test with Float64
        output_dir_f64 = mktempdir()

        try
            sim_f64 = run_mvmc_from_stdface(fixture_path; T=Float64, output_dir=output_dir_f64)
            @test sim_f64 isa ManyVariableVariationalMonteCarlo.EnhancedVMCSimulation{Float64}

        finally
            rm(output_dir_f64, recursive=true, force=true)
        end

        # Test with ComplexF64
        output_dir_cf64 = mktempdir()

        try
            sim_cf64 = run_mvmc_from_stdface(fixture_path; T=ComplexF64, output_dir=output_dir_cf64)
            @test sim_cf64 isa ManyVariableVariationalMonteCarlo.EnhancedVMCSimulation{ComplexF64}

        finally
            rm(output_dir_cf64, recursive=true, force=true)
        end
    else
        @test_skip "Test fixture file not available"
    end
end

@testitem "print_simulation_summary functionality" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    fixture_path = joinpath(@__DIR__, "fixtures", "simple_spin_chain.def")

    if isfile(fixture_path)
        output_dir = mktempdir()

        try
            sim = run_mvmc_from_stdface(fixture_path; T=ComplexF64, output_dir=output_dir)

            # Test summary printing
            output_str = let
                temp_file = tempname()
                open(temp_file, "w") do io
                    redirect_stdout(io) do
                        print_simulation_summary(sim)
                    end
                end
                result = read(temp_file, String)
                rm(temp_file, force=true)
                result
            end

            @test contains(output_str, "VMC Enhanced Simulation Results")
            @test contains(output_str, "Mode:")
            @test contains(output_str, "Sites:")

            # Should contain either optimization or physics results
            @test contains(output_str, "Energy") || contains(output_str, "Samples")

        finally
            rm(output_dir, recursive=true, force=true)
        end
    else
        @test_skip "Test fixture file not available"
    end
end

@testitem "run_mvmc_from_stdface with real mVMC sample" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    # Try to use real mVMC sample file if available
    pkgroot = pkgdir(ManyVariableVariationalMonteCarlo)
    sample_path = joinpath(pkgroot, "mVMC", "samples", "Standard", "Spin", "HeisenbergChain", "StdFace.def")

    if isfile(sample_path)
        # Create a modified version with smaller parameters for faster testing
        params = parse_stdface_def(sample_path)

        # Reduce system size and sampling for fast test
        params.L = 4
        params.NSROptItrStep = 5
        params.NVMCWarmUp = 3
        params.NVMCSample = 10

        # Create temporary file with modified parameters
        temp_content = """
        L = $(params.L)
        model = "$(params.model)"
        lattice = "$(params.lattice)"
        J = $(params.J)
        2Sz = $(params.TwoSz)
        NVMCCalMode = 1
        NSROptItrStep = 5
        NVMCWarmUp = 3
        NVMCSample = 10
        RndSeed = 12345
        """

        temp_file = tempname() * "_mVMC_sample.def"
        write(temp_file, temp_content)
        output_dir = mktempdir()

        try
            sim = run_mvmc_from_stdface(temp_file; T=ComplexF64, output_dir=output_dir)

            @test sim.config.model == :Spin
            @test sim.config.lattice == :chain
            @test sim.config.nsites == 4

            # Should complete without errors
            @test sim.vmc_state !== nothing
            @test sim.quantum_projection !== nothing

        finally
            rm(temp_file, force=true)
            rm(output_dir, recursive=true, force=true)
        end
    else
        @test_skip "mVMC sample file not available"
    end
end

@testitem "run_mvmc_from_stdface workflow components integration" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    fixture_path = joinpath(@__DIR__, "fixtures", "simple_spin_chain.def")

    if isfile(fixture_path)
        output_dir = mktempdir()

        try
            # Test individual workflow components

            # 1. Parse StdFace.def
            params = parse_stdface_def(fixture_path)
            @test params.L == 4
            @test params.model == "Spin"

            # 2. Generate expert mode files
            ManyVariableVariationalMonteCarlo.generate_expert_mode_files(params, output_dir)
            @test isfile(joinpath(output_dir, "namelist.def"))

            # 3. Create config
            config = ManyVariableVariationalMonteCarlo.stdface_to_simulation_config(params; root=dirname(fixture_path))
            @test config.nsites == 4

            # 4. Create parameter layout
            layout = ManyVariableVariationalMonteCarlo.create_parameter_layout(config)
            @test layout.nproj > 0

            # 5. Create simulation
            sim = ManyVariableVariationalMonteCarlo.EnhancedVMCSimulation{ComplexF64}(config, layout)
            @test sim isa ManyVariableVariationalMonteCarlo.EnhancedVMCSimulation

            # 6. Initialize simulation
            ManyVariableVariationalMonteCarlo.initialize_enhanced_simulation!(sim)
            @test sim.vmc_state !== nothing

            # 7. Initialize quantum projection
            sim.quantum_projection = initialize_quantum_projection_from_config(config; T=ComplexF64)
            @test sim.quantum_projection !== nothing

            # All components should work together
            @test sim.mode == PHYSICS_CALCULATION

        finally
            rm(output_dir, recursive=true, force=true)
        end
    else
        @test_skip "Test fixture file not available"
    end
end

@testitem "run_mvmc_from_stdface memory and performance" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    # Test with slightly larger system to check memory handling
    test_content = """
    L = 8
    model = "Spin"
    lattice = "chain"
    J = 1.0
    2Sz = 0
    NVMCCalMode = 1
    NSROptItrStep = 3
    NVMCWarmUp = 2
    NVMCSample = 5
    RndSeed = 12345
    """

    temp_file = tempname() * "_perf.def"
    write(temp_file, test_content)
    output_dir = mktempdir()

    try
        # Measure basic performance
        start_time = time()
        sim = run_mvmc_from_stdface(temp_file; T=ComplexF64, output_dir=output_dir)
        elapsed_time = time() - start_time

        @test sim.config.nsites == 8
        @test elapsed_time < 60.0  # Should complete within reasonable time

        # Check memory usage is reasonable
        @test sim.vmc_state.n_sites == 8
        @test sim.vmc_state.n_electrons == 8

    finally
        rm(temp_file, force=true)
        rm(output_dir, recursive=true, force=true)
    end
end
