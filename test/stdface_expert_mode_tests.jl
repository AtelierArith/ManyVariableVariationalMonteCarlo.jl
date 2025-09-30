# Expert Mode File Generation Tests
# Tests for StdFace expert mode file generation functionality

@testitem "generate_expert_mode_files basic functionality" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    # Create test parameters
    params = ManyVariableVariationalMonteCarlo.StdFaceParameters()
    params.L = 4
    params.model = "Spin"
    params.lattice = "chain"
    params.J = 1.0
    params.TwoSz = 0

    # Create temporary output directory
    output_dir = mktempdir()

    try
        # Test file generation
        ManyVariableVariationalMonteCarlo.generate_expert_mode_files(params, output_dir)

        # Check that required files are created
        @test isfile(joinpath(output_dir, "locspn.def"))
        @test isfile(joinpath(output_dir, "trans.def"))
        @test isfile(joinpath(output_dir, "modpara.def"))
        @test isfile(joinpath(output_dir, "namelist.def"))
        @test isfile(joinpath(output_dir, "gutzwilleridx.def"))
        @test isfile(joinpath(output_dir, "jastrowidx.def"))
        @test isfile(joinpath(output_dir, "greenone.def"))

        # Check that files are not empty
        @test filesize(joinpath(output_dir, "locspn.def")) > 0
        @test filesize(joinpath(output_dir, "trans.def")) > 0
        @test filesize(joinpath(output_dir, "modpara.def")) > 0
        @test filesize(joinpath(output_dir, "namelist.def")) > 0

    finally
        rm(output_dir, recursive=true, force=true)
    end
end

@testitem "create_lattice_data function" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    # Test chain lattice
    params = ManyVariableVariationalMonteCarlo.StdFaceParameters()
    params.L = 4
    params.model = "Spin"
    params.lattice = "chain"

    lattice_data = ManyVariableVariationalMonteCarlo.create_lattice_data(params)

    @test lattice_data.nsite == 4
    @test lattice_data.nsiteuc == 1  # Chain has 1 site per unit cell
    @test length(lattice_data.locspinflag) == 4

    # For spin model, all sites should have local spin flag = 1
    @test all(flag -> flag == 1, lattice_data.locspinflag)
end

@testitem "create_model_data function" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    # Test Spin chain model
    params = ManyVariableVariationalMonteCarlo.StdFaceParameters()
    params.L = 4
    params.model = "Spin"
    params.lattice = "chain"
    params.J = 1.5

    lattice_data = ManyVariableVariationalMonteCarlo.create_lattice_data(params)
    model_data = ManyVariableVariationalMonteCarlo.create_model_data(params, lattice_data)

    @test model_data.lattice === lattice_data
    @test model_data.nhund > 0  # Should have Hund terms for spin model
    @test length(model_data.hundindx) == model_data.nhund
    @test length(model_data.hund) == model_data.nhund

    # Check that Hund coupling values are correct
    # Note: Hund coupling is -J/2.0 according to C implementation
    for hund_val in model_data.hund
        @test hund_val == -params.J / 2.0  # Should be -J/2.0
    end
end

@testitem "locspn.def file content validation" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    params = ManyVariableVariationalMonteCarlo.StdFaceParameters()
    params.L = 3
    params.model = "Spin"
    params.lattice = "chain"

    output_dir = mktempdir()

    try
        lattice_data = ManyVariableVariationalMonteCarlo.create_lattice_data(params)
        model_data = ManyVariableVariationalMonteCarlo.create_model_data(params, lattice_data)
        ManyVariableVariationalMonteCarlo.print_locspn_def(model_data, output_dir)

        locspn_file = joinpath(output_dir, "locspn.def")
        @test isfile(locspn_file)

        content = read(locspn_file, String)
        lines = split(content, '\n')

        # Check header format
        @test any(line -> contains(line, "NlocalSpin"), lines)
        @test any(line -> contains(line, "3"), lines)  # Should show 3 local spins

        # Check site entries (0-based indexing in file)
        @test any(line -> contains(line, "0") && contains(line, "1"), lines)
        @test any(line -> contains(line, "1") && contains(line, "1"), lines)
        @test any(line -> contains(line, "2") && contains(line, "1"), lines)

    finally
        rm(output_dir, recursive=true, force=true)
    end
end

@testitem "trans.def file content validation" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    params = ManyVariableVariationalMonteCarlo.StdFaceParameters()
    params.L = 3
    params.model = "Hubbard"
    params.lattice = "chain"
    params.t = 1.0

    output_dir = mktempdir()

    try
        lattice_data = ManyVariableVariationalMonteCarlo.create_lattice_data(params)
        model_data = ManyVariableVariationalMonteCarlo.create_model_data(params, lattice_data)
        ManyVariableVariationalMonteCarlo.print_trans_def(model_data, output_dir)

        trans_file = joinpath(output_dir, "trans.def")
        @test isfile(trans_file)

        content = read(trans_file, String)
        lines = split(content, '\n')

        # Check header format
        @test any(line -> contains(line, "NTransfer"), lines)

        # Should have transfer terms (hopping)
        @test any(line -> contains(line, "-1.000000"), lines)  # -t hopping

    finally
        rm(output_dir, recursive=true, force=true)
    end
end

@testitem "modpara.def file content validation" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    params = ManyVariableVariationalMonteCarlo.StdFaceParameters()
    params.L = 4
    params.NSROptItrStep = 100
    params.NVMCWarmUp = 10
    params.NVMCSample = 50
    params.RndSeed = 12345

    output_dir = mktempdir()

    try
        ManyVariableVariationalMonteCarlo.print_modpara_def(params, output_dir)

        modpara_file = joinpath(output_dir, "modpara.def")
        @test isfile(modpara_file)

        content = read(modpara_file, String)
        lines = split(content, '\n')

        # Check that key parameters are present
        @test any(line -> contains(line, "NSROptItrStep") && contains(line, "100"), lines)
        @test any(line -> contains(line, "NVMCWarmUp") && contains(line, "10"), lines)
        @test any(line -> contains(line, "NVMCSample") && contains(line, "50"), lines)
        @test any(line -> contains(line, "RndSeed") && contains(line, "12345"), lines)

    finally
        rm(output_dir, recursive=true, force=true)
    end
end

@testitem "namelist.def file content validation" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    params = ManyVariableVariationalMonteCarlo.StdFaceParameters()
    params.L = 4
    params.model = "Spin"
    params.lattice = "chain"

    output_dir = mktempdir()

    try
        lattice_data = ManyVariableVariationalMonteCarlo.create_lattice_data(params)
        model_data = ManyVariableVariationalMonteCarlo.create_model_data(params, lattice_data)
        ManyVariableVariationalMonteCarlo.print_namelist_def(params, model_data, output_dir)

        namelist_file = joinpath(output_dir, "namelist.def")
        @test isfile(namelist_file)

        content = read(namelist_file, String)
        lines = split(content, '\n')

        # Check that required def files are listed
        @test any(line -> contains(line, "modpara.def"), lines)
        @test any(line -> contains(line, "locspn.def"), lines)
        @test any(line -> contains(line, "greenone.def"), lines)

        # For spin model, should include Hund terms
        @test any(line -> contains(line, "hund.def"), lines)

    finally
        rm(output_dir, recursive=true, force=true)
    end
end

@testitem "expert mode files for different models" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    # Test Hubbard model
    params_hubbard = ManyVariableVariationalMonteCarlo.StdFaceParameters()
    params_hubbard.L = 3
    params_hubbard.model = "Hubbard"
    params_hubbard.lattice = "chain"
    params_hubbard.t = 1.0
    params_hubbard.U = 2.0

    output_dir_hubbard = mktempdir()

    try
        ManyVariableVariationalMonteCarlo.generate_expert_mode_files(params_hubbard, output_dir_hubbard)

        # Hubbard model should have transfer terms
        @test isfile(joinpath(output_dir_hubbard, "trans.def"))
        trans_content = read(joinpath(output_dir_hubbard, "trans.def"), String)
        @test contains(trans_content, "NTransfer")

        # Should have Coulomb intra terms
        @test isfile(joinpath(output_dir_hubbard, "coulombintra.def"))

    finally
        rm(output_dir_hubbard, recursive=true, force=true)
    end

    # Test Spin model
    params_spin = ManyVariableVariationalMonteCarlo.StdFaceParameters()
    params_spin.L = 3
    params_spin.model = "Spin"
    params_spin.lattice = "chain"
    params_spin.J = 1.0

    output_dir_spin = mktempdir()

    try
        ManyVariableVariationalMonteCarlo.generate_expert_mode_files(params_spin, output_dir_spin)

        # Spin model should have Hund terms
        @test isfile(joinpath(output_dir_spin, "hund.def"))
        hund_content = read(joinpath(output_dir_spin, "hund.def"), String)
        @test contains(hund_content, "NHund")

    finally
        rm(output_dir_spin, recursive=true, force=true)
    end
end

@testitem "expert mode file generation with test fixtures" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    # Test with fixture files
    fixture_dir = joinpath(@__DIR__, "fixtures")

    if isfile(joinpath(fixture_dir, "simple_spin_chain.def"))
        params = parse_stdface_def(joinpath(fixture_dir, "simple_spin_chain.def"))
        output_dir = mktempdir()

        try
            ManyVariableVariationalMonteCarlo.generate_expert_mode_files(params, output_dir)

            # Verify all expected files are generated
            expected_files = [
                "locspn.def", "trans.def", "modpara.def", "namelist.def",
                "gutzwilleridx.def", "jastrowidx.def", "greenone.def", "hund.def"
            ]

            for file in expected_files
                @test isfile(joinpath(output_dir, file))
                @test filesize(joinpath(output_dir, file)) > 0
            end

        finally
            rm(output_dir, recursive=true, force=true)
        end
    else
        @test_skip "Test fixture file not available"
    end
end

@testitem "error handling in expert mode file generation" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    params = ManyVariableVariationalMonteCarlo.StdFaceParameters()
    params.L = 4

    # Test with invalid output directory (read-only)
    if !Sys.iswindows()  # Skip on Windows due to permission model differences
        readonly_dir = mktempdir()
        chmod(readonly_dir, 0o444)  # Read-only

        try
            # Should handle permission errors gracefully
            @test_throws Exception ManyVariableVariationalMonteCarlo.generate_expert_mode_files(params, readonly_dir)
        finally
            chmod(readonly_dir, 0o755)  # Restore permissions for cleanup
            rm(readonly_dir, recursive=true, force=true)
        end
    end
end
