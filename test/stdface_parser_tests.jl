# StdFace Parser Tests
# Tests for StdFace.def parsing functionality and related functions

@testitem "parse_stdface_def basic functionality" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    # Create a temporary StdFace.def file for testing
    test_content = """
    L = 8
    W = 1
    model = "Spin"
    lattice = "chain"
    J = 1.0
    2Sz = 0
    NSROptItrStep = 100
    NVMCWarmUp = 10
    NVMCSample = 100
    """

    temp_file = tempname() * "_StdFace.def"
    write(temp_file, test_content)

    try
        # Test successful parsing
        params = parse_stdface_def(temp_file)

        @test params.L == 8
        @test params.W == 1
        @test params.model == "Spin"
        @test params.lattice == "chain"
        @test params.J == 1.0
        @test params.TwoSz == 0
        @test params.NSROptItrStep == 100
        @test params.NVMCWarmUp == 10
        @test params.NVMCSample == 100

        # Test default values are preserved
        @test params.Height == 1  # Default value
        @test params.U == 4.0     # Default value
        @test params.t == 1.0     # Default value

    finally
        rm(temp_file, force=true)
    end
end

@testitem "parse_stdface_def error handling" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    # Test file not found error
    @test_throws Exception parse_stdface_def("nonexistent_file.def")

    # Test with empty file
    temp_file = tempname() * "_empty.def"
    write(temp_file, "")

    try
        params = parse_stdface_def(temp_file)
        # Should use default values
        @test params.L == 8      # Default
        @test params.model == "Hubbard"  # Default
        @test params.lattice == "square" # Default
    finally
        rm(temp_file, force=true)
    end
end

@testitem "parse_parameter! individual parameter parsing" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    params = ManyVariableVariationalMonteCarlo.StdFaceParameters()

    # Test integer parameter parsing
    ManyVariableVariationalMonteCarlo.parse_parameter!(params, "L", "16")
    @test params.L == 16

    # Test float parameter parsing
    ManyVariableVariationalMonteCarlo.parse_parameter!(params, "J", "2.5")
    @test params.J == 2.5

    # Test string parameter parsing
    ManyVariableVariationalMonteCarlo.parse_parameter!(params, "model", "Hubbard")
    @test params.model == "Hubbard"

    # Test quoted string parameter parsing
    ManyVariableVariationalMonteCarlo.parse_parameter!(params, "lattice", "\"triangular\"")
    @test params.lattice == "triangular"

    # Test case insensitive parsing
    ManyVariableVariationalMonteCarlo.parse_parameter!(params, "w", "3")
    @test params.W == 3

    ManyVariableVariationalMonteCarlo.parse_parameter!(params, "MODEL", "Spin")
    @test params.model == "Spin"
end

@testitem "parse_stdface_def comment and empty line handling" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    test_content = """
    # This is a comment
    L = 4
    // Another comment style
    W = 2

    # Empty line above and below

    model = "Hubbard"
    lattice = "square"  # Inline comment
    """

    temp_file = tempname() * "_comments.def"
    write(temp_file, test_content)

    try
        params = parse_stdface_def(temp_file)

        @test params.L == 4
        @test params.W == 2
        @test params.model == "Hubbard"
        @test params.lattice == "square"

    finally
        rm(temp_file, force=true)
    end
end

@testitem "print_stdface_summary output format" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    # Create test parameters for Spin model
    params = ManyVariableVariationalMonteCarlo.StdFaceParameters()
    params.L = 8
    params.model = "Spin"
    params.lattice = "chain"
    params.J = 1.5
    params.TwoSz = 0
    params.NSROptItrStep = 200

    # Capture output by redirecting to a temp file
    output_str = let
        temp_file = tempname()
        open(temp_file, "w") do io
            redirect_stdout(io) do
                print_stdface_summary(params)
            end
        end
        result = read(temp_file, String)
        rm(temp_file, force=true)
        result
    end

    # Check that key information is present
    @test contains(output_str, "Input Parameter of Standard Interface")
    @test contains(output_str, "L                    | VALUE : 8")
    @test contains(output_str, "model                | VALUE : Spin")
    @test contains(output_str, "lattice              | VALUE : chain")
    @test contains(output_str, "J                    | VALUE : 1.5")
    @test contains(output_str, "NSROptItrStep        | VALUE : 200")
    @test contains(output_str, "2Sz                  | VALUE : 0")
    @test contains(output_str, "Construct Model")
    @test contains(output_str, "Lattice Size & Shape")
    @test contains(output_str, "Number of Cell = 8")
    @test contains(output_str, "Hamiltonian")

    # Test Hubbard model output
    params.model = "Hubbard"
    params.t = 1.0
    params.U = 4.0

    output_str = let
        temp_file = tempname()
        open(temp_file, "w") do io
            redirect_stdout(io) do
                print_stdface_summary(params)
            end
        end
        result = read(temp_file, String)
        rm(temp_file, force=true)
        result
    end

    @test contains(output_str, "t                    | VALUE : 1.0")
    @test contains(output_str, "U                    | VALUE : 4.0")
end

@testitem "stdface_to_simulation_config conversion" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    # Create test parameters
    params = ManyVariableVariationalMonteCarlo.StdFaceParameters()
    params.L = 6
    params.W = 1
    params.model = "Spin"
    params.lattice = "chain"
    params.J = 1.0
    params.TwoSz = 0
    params.NSROptItrStep = 100
    params.NVMCWarmUp = 10
    params.NVMCSample = 50
    params.NVMCCalMode = 1
    params.CDataFileHead = "test_output"
    params.OneBodyG = true
    params.TwoBodyG = false

    # Test conversion
    config = ManyVariableVariationalMonteCarlo.stdface_to_simulation_config(params; root = ".")

    @test config isa SimulationConfig
    @test facevalue(config.face, :L, Int) == 6
    @test facevalue(config.face, :W, Int) == 1
    @test facevalue(config.face, :model, String) == "Spin"
    @test facevalue(config.face, :lattice, String) == "chain"
    @test facevalue(config.face, :J, Float64) == 1.0
    @test facevalue(config.face, :TwoSz, Int) == 0
    @test facevalue(config.face, :NSROptItrStep, Int) == 100
    @test facevalue(config.face, :NVMCWarmUp, Int) == 10
    @test facevalue(config.face, :NVMCSample, Int) == 50
    @test facevalue(config.face, :NVMCCalMode, Int) == 1
    @test facevalue(config.face, :CDataFileHead, String) == "test_output"
    @test facevalue(config.face, :OneBodyG, Bool) == true
    @test facevalue(config.face, :TwoBodyG, Bool) == false
    @test facevalue(config.face, :StdFaceRoot, String) == "."
end

@testitem "parse_stdface_and_create_config convenience function" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    # Create a temporary StdFace.def file
    test_content = """
    L = 4
    model = "Spin"
    lattice = "chain"
    J = 2.0
    NVMCCalMode = 0
    """

    temp_dir = mktempdir()
    temp_file = joinpath(temp_dir, "StdFace.def")
    write(temp_file, test_content)

    try
        # Test convenience function
        config = parse_stdface_and_create_config(temp_file)

        @test config isa SimulationConfig
        @test facevalue(config.face, :L, Int) == 4
        @test facevalue(config.face, :model, String) == "Spin"
        @test facevalue(config.face, :lattice, String) == "chain"
        @test facevalue(config.face, :J, Float64) == 2.0
        @test facevalue(config.face, :NVMCCalMode, Int) == 0
        @test facevalue(config.face, :StdFaceRoot, String) == temp_dir

    finally
        rm(temp_dir, recursive=true, force=true)
    end
end

@testitem "StdFaceParameters default constructor" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    params = ManyVariableVariationalMonteCarlo.StdFaceParameters()

    # Test default values match expected defaults
    @test params.L == 8
    @test params.W == 1
    @test params.Height == 1
    @test params.model == "Hubbard"
    @test params.lattice == "square"
    @test params.J == 1.0
    @test params.t == 1.0
    @test params.U == 4.0
    @test params.TwoSz == 0
    @test params.TwoS == 1
    @test params.NSROptItrStep == 1000
    @test params.NSROptItrSmp == 100
    @test params.NVMCWarmUp == 10
    @test params.NVMCSample == 10
    @test params.RndSeed == 11272  # C implementation default
    @test params.CDataFileHead == "zvo"
    @test params.CParaFileHead == "zqp"
    @test params.OneBodyG == true
    @test params.TwoBodyG == false
end

@testitem "parse_stdface_def with real mVMC sample file" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    # Try to use real mVMC sample file if available
    pkgroot = pkgdir(ManyVariableVariationalMonteCarlo)
    sample_path = joinpath(pkgroot, "mVMC", "samples", "Standard", "Spin", "HeisenbergChain", "StdFace.def")

    if isfile(sample_path)
        params = parse_stdface_def(sample_path)

        # Test that we can parse the real file without errors
        @test params isa ManyVariableVariationalMonteCarlo.StdFaceParameters
        @test params.model == "Spin"
        @test params.lattice == "chain"
        @test params.L > 0
        @test params.J != 0.0  # Should have some exchange coupling

        # Test conversion to SimulationConfig
        config = ManyVariableVariationalMonteCarlo.stdface_to_simulation_config(params; root = dirname(sample_path))
        @test config isa SimulationConfig
        @test config.model == :Spin
        @test config.lattice == :chain
    else
        @test_skip "mVMC sample file not available"
    end
end
