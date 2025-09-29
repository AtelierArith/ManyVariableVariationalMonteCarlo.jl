"""
StdFace.def Parser

Parser for mVMC StdFace.def format input files.
Converts StdFace parameters to SimulationConfig compatible format.

Based on the C reference implementation in mVMC/src/StdFace/
"""

using Printf

"""
    StdFaceParameters

Container for parameters parsed from StdFace.def files.
"""
mutable struct StdFaceParameters
    # Lattice parameters
    L::Int
    W::Int
    Height::Int
    Lsub::Int
    Wsub::Int
    Hsub::Int

    # Model specification
    model::String
    lattice::String

    # Physical parameters
    J::Float64
    J0x::Float64
    J0y::Float64
    J0z::Float64
    J1::Float64
    J2::Float64
    h::Float64
    Gamma::Float64
    D::Float64
    t::Float64
    t0::Float64
    t1::Float64
    t2::Float64
    U::Float64
    V::Float64
    V0::Float64
    V1::Float64
    V2::Float64
    mu::Float64

    # Spin and symmetry
    TwoSz::Int
    TwoS::Int
    phase0::Float64
    phase1::Float64

    # Lattice geometry
    a::Float64
    Wlength::Float64
    Llength::Float64
    Wx::Float64
    Wy::Float64
    Lx::Float64
    Ly::Float64

    # VMC calculation parameters
    NVMCCalMode::Int
    NSROptItrStep::Int
    NSROptItrSmp::Int
    NVMCWarmUp::Int
    NVMCInterval::Int
    NVMCSample::Int
    DSROptRedCut::Float64
    DSROptStaDel::Float64
    DSROptStepDt::Float64
    RndSeed::Int
    NSplitSize::Int
    NSRCG::Int
    NSROptCGMaxIter::Int
    DSROptCGTol::Float64

    # Lanczos parameters
    NLanczosMode::Int

    # Data output parameters
    NDataIdxStart::Int
    NDataQtySmp::Int
    CDataFileHead::String
    CParaFileHead::String

    # Numerical parameters
    NSPGaussLeg::Int
    NSPStot::Int
    NMPTrans::Int
    ComplexType::Int
    NStore::Int
    NStoreO::Int

    # Output control
    ioutputmode::Int
    OneBodyG::Union{Bool,String}
    OneBodyGMode::String
    TwoBodyG::Bool

    # File flush control
    FlushFile::Bool
    NFileFlushInterval::Int

    function StdFaceParameters()
        new(
            # Lattice parameters - defaults
            8, 1, 1, 2, 1, 1,
            # Model
            "Hubbard", "square",
            # Physical parameters - defaults
            1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            # Spin and symmetry
            0, 1, 0.0, 0.0,
            # Lattice geometry
            1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0,
            # VMC parameters (matching C implementation defaults)
            0, 1000, 30, 10, 1, 1000, 0.001, 0.02, 0.02, 123456789, 1, 0, 100, 1e-10,
            # Lanczos
            0,
            # Data output
            1, 1, "zvo", "zqp",
            # Numerical
            8, 0, -1, 0, 1, 1,
            # Output control
            1, true, "local", false,
            # File flush
            false, 0
        )
    end
end

"""
    parse_stdface_def(filename::String) -> StdFaceParameters

Parse a StdFace.def format file and return a StdFaceParameters object.
"""
function parse_stdface_def(filename::String)
    params = StdFaceParameters()

    if !isfile(filename)
        error("StdFace.def file not found: $filename")
    end

    open(filename, "r") do file
        for line in eachline(file)
            line = strip(line)

            # Skip comments and empty lines
            if isempty(line) || startswith(line, "//") || startswith(line, "#")
                continue
            end

            # Parse key-value pairs
            if contains(line, "=")
                key, value = split(line, "=", limit=2)
                key = strip(key)
                value = strip(value)

                # Remove quotes from string values
                if startswith(value, "\"") && endswith(value, "\"")
                    value = value[2:end-1]
                end

                # Parse based on key
                parse_parameter!(params, key, value)
            end
        end
    end

    return params
end

"""
    parse_parameter!(params::StdFaceParameters, key::AbstractString, value::AbstractString)

Parse a single parameter and update the StdFaceParameters object.
"""
function parse_parameter!(params::StdFaceParameters, key::AbstractString, value::AbstractString)
    # Convert key to lowercase for case-insensitive matching
    key_lower = lowercase(key)

    try
        if key_lower == "l"
            params.L = parse(Int, value)
        elseif key_lower == "w"
            params.W = parse(Int, value)
        elseif key_lower == "height"
            params.Height = parse(Int, value)
        elseif key_lower == "lsub"
            params.Lsub = parse(Int, value)
        elseif key_lower == "wsub"
            params.Wsub = parse(Int, value)
        elseif key_lower == "hsub"
            params.Hsub = parse(Int, value)
        elseif key_lower == "model"
            params.model = value
        elseif key_lower == "lattice"
            params.lattice = value
        elseif key_lower == "j"
            params.J = parse(Float64, value)
        elseif key_lower == "j0x"
            params.J0x = parse(Float64, value)
        elseif key_lower == "j0y"
            params.J0y = parse(Float64, value)
        elseif key_lower == "j0z"
            params.J0z = parse(Float64, value)
        elseif key_lower == "j1"
            params.J1 = parse(Float64, value)
        elseif key_lower == "j2"
            params.J2 = parse(Float64, value)
        elseif key_lower == "h"
            params.h = parse(Float64, value)
        elseif key_lower == "gamma"
            params.Gamma = parse(Float64, value)
        elseif key_lower == "d"
            params.D = parse(Float64, value)
        elseif key_lower == "t"
            params.t = parse(Float64, value)
        elseif key_lower == "t0"
            params.t0 = parse(Float64, value)
        elseif key_lower == "t1"
            params.t1 = parse(Float64, value)
        elseif key_lower == "t2"
            params.t2 = parse(Float64, value)
        elseif key_lower == "u"
            params.U = parse(Float64, value)
        elseif key_lower == "v"
            params.V = parse(Float64, value)
        elseif key_lower == "v0"
            params.V0 = parse(Float64, value)
        elseif key_lower == "v1"
            params.V1 = parse(Float64, value)
        elseif key_lower == "v2"
            params.V2 = parse(Float64, value)
        elseif key_lower == "mu"
            params.mu = parse(Float64, value)
        elseif key_lower == "2sz"
            params.TwoSz = parse(Int, value)
        elseif key_lower == "2s"
            params.TwoS = parse(Int, value)
        elseif key_lower == "phase0"
            params.phase0 = parse(Float64, value)
        elseif key_lower == "phase1"
            params.phase1 = parse(Float64, value)
        elseif key_lower == "a"
            params.a = parse(Float64, value)
        elseif key_lower == "wlength"
            params.Wlength = parse(Float64, value)
        elseif key_lower == "llength"
            params.Llength = parse(Float64, value)
        elseif key_lower == "wx"
            params.Wx = parse(Float64, value)
        elseif key_lower == "wy"
            params.Wy = parse(Float64, value)
        elseif key_lower == "lx"
            params.Lx = parse(Float64, value)
        elseif key_lower == "ly"
            params.Ly = parse(Float64, value)
        elseif key_lower == "nvmccalmode"
            params.NVMCCalMode = parse(Int, value)
        elseif key_lower == "nsroptitrstep"
            params.NSROptItrStep = parse(Int, value)
        elseif key_lower == "nsroptitrsmp"
            params.NSROptItrSmp = parse(Int, value)
        elseif key_lower == "nvmcwarmup"
            params.NVMCWarmUp = parse(Int, value)
        elseif key_lower == "nvmcinterval"
            params.NVMCInterval = parse(Int, value)
        elseif key_lower == "nvmcsample"
            params.NVMCSample = parse(Int, value)
        elseif key_lower == "dsroptredcut"
            params.DSROptRedCut = parse(Float64, value)
        elseif key_lower == "dsroptstawddel"
            params.DSROptStaDel = parse(Float64, value)
        elseif key_lower == "dsroptstepdt"
            params.DSROptStepDt = parse(Float64, value)
        elseif key_lower == "rndseed"
            params.RndSeed = parse(Int, value)
        elseif key_lower == "nsplitsize"
            params.NSplitSize = parse(Int, value)
        elseif key_lower == "nsrcg"
            params.NSRCG = parse(Int, value)
        elseif key_lower == "nsroptcgmaxiter"
            params.NSROptCGMaxIter = parse(Int, value)
        elseif key_lower == "dsroptcgtol"
            params.DSROptCGTol = parse(Float64, value)
        elseif key_lower == "nlanczosmode"
            params.NLanczosMode = parse(Int, value)
        elseif key_lower == "ndataidxstart"
            params.NDataIdxStart = parse(Int, value)
        elseif key_lower == "ndataqtysmp"
            params.NDataQtySmp = parse(Int, value)
        elseif key_lower == "cdatafilehead"
            params.CDataFileHead = value
        elseif key_lower == "cparafilehead"
            params.CParaFileHead = value
        elseif key_lower == "nspgaussleg"
            params.NSPGaussLeg = parse(Int, value)
        elseif key_lower == "nspstot"
            params.NSPStot = parse(Int, value)
        elseif key_lower == "nmptrans"
            params.NMPTrans = parse(Int, value)
        elseif key_lower == "complextype"
            params.ComplexType = parse(Int, value)
        elseif key_lower == "nstore"
            params.NStore = parse(Int, value)
        elseif key_lower == "nstoreo"
            params.NStoreO = parse(Int, value)
        elseif key_lower == "ioutputmode"
            params.ioutputmode = parse(Int, value)
        elseif key_lower == "onebodyg"
            # Handle both boolean and string values
            value_lower = lowercase(value)
            if value_lower in ["true", "1"]
                params.OneBodyG = true
            elseif value_lower in ["false", "0"]
                params.OneBodyG = false
            else
                params.OneBodyG = value
            end
        elseif key_lower == "onebodygmode"
            params.OneBodyGMode = value
        elseif key_lower == "twobodyg"
            value_lower = lowercase(value)
            params.TwoBodyG = value_lower in ["true", "1"]
        elseif key_lower == "flushfile"
            value_lower = lowercase(value)
            params.FlushFile = value_lower in ["true", "1"]
        elseif key_lower == "nfileflushinterval"
            params.NFileFlushInterval = parse(Int, value)
        else
            @debug "Unknown StdFace parameter: $key = $value"
        end
    catch e
        @warn "Failed to parse parameter $key = $value: $e"
    end
end

"""
    stdface_to_simulation_config(params::StdFaceParameters) -> SimulationConfig

Convert StdFaceParameters to a SimulationConfig object.
"""
function stdface_to_simulation_config(params::StdFaceParameters; root::AbstractString = ".")
    # Create face dictionary with all parameters
    face = FaceDefinition()

    # Add parameters to face definition
    push_definition!(face, :L, params.L)
    push_definition!(face, :W, params.W)
    push_definition!(face, :Height, params.Height)
    push_definition!(face, :Lsub, params.Lsub)
    push_definition!(face, :Wsub, params.Wsub)
    push_definition!(face, :Hsub, params.Hsub)
    push_definition!(face, :model, params.model)
    push_definition!(face, :lattice, params.lattice)
    push_definition!(face, :J, params.J)
    push_definition!(face, :J0x, params.J0x)
    push_definition!(face, :J0y, params.J0y)
    push_definition!(face, :J0z, params.J0z)
    push_definition!(face, :t, params.t)
    push_definition!(face, :U, params.U)
    push_definition!(face, :TwoSz, params.TwoSz)
    push_definition!(face, :NVMCCalMode, params.NVMCCalMode)
    push_definition!(face, :NSROptItrStep, params.NSROptItrStep)
    push_definition!(face, :NSROptItrSmp, params.NSROptItrSmp)
    push_definition!(face, :NVMCWarmUp, params.NVMCWarmUp)
    push_definition!(face, :NVMCInterval, params.NVMCInterval)
    push_definition!(face, :NVMCSample, params.NVMCSample)
    push_definition!(face, :DSROptRedCut, params.DSROptRedCut)
    push_definition!(face, :DSROptStaDel, params.DSROptStaDel)
    push_definition!(face, :DSROptStepDt, params.DSROptStepDt)
    push_definition!(face, :RndSeed, params.RndSeed)
    push_definition!(face, :NLanczosMode, params.NLanczosMode)
    push_definition!(face, :CDataFileHead, params.CDataFileHead)
    push_definition!(face, :CParaFileHead, params.CParaFileHead)
    push_definition!(face, :OneBodyG, params.OneBodyG)
    push_definition!(face, :TwoBodyG, params.TwoBodyG)

    # Store StdFace root for downstream file lookups
    push_definition!(face, :StdFaceRoot, String(root))

    # Create SimulationConfig using the standard constructor
    config = SimulationConfig(face; root = String(root))

    return config
end

"""
    parse_stdface_and_create_config(filename::String) -> SimulationConfig

Convenience function to parse StdFace.def file and create SimulationConfig.
"""
function parse_stdface_and_create_config(filename::String)
    params = parse_stdface_def(filename)
    return stdface_to_simulation_config(params; root = dirname(filename))
end

"""
    print_stdface_summary(params::StdFaceParameters)

Print a summary of parsed StdFace parameters (similar to C implementation output).
"""
function print_stdface_summary(params::StdFaceParameters)
    println("######  Input Parameter of Standard Interface  ######")
    println()
    println("  Open Standard-Mode Inputfile StdFace.def")
    println()

    # Print key parameters
    println("  KEYWORD : L                    | VALUE : $(params.L)")
    println("  KEYWORD : Lsub                 | VALUE : $(params.Lsub)")
    println("  KEYWORD : model                | VALUE : $(params.model)")
    println("  KEYWORD : lattice              | VALUE : $(params.lattice)")

    if params.model == "Spin"
        println("  KEYWORD : J                    | VALUE : $(params.J)")
    else
        println("  KEYWORD : t                    | VALUE : $(params.t)")
        println("  KEYWORD : U                    | VALUE : $(params.U)")
    end

    println("  KEYWORD : NSROptItrStep        | VALUE : $(params.NSROptItrStep)")
    println("  KEYWORD : 2Sz                  | VALUE : $(params.TwoSz)")
    println()

    println("#######  Construct Model  #######")
    println()
    println("  @ Lattice Size & Shape")
    println()
    println("    L = $(params.L)")
    println("    W = $(params.W)")
    println("    Height = $(params.Height)")
    println("    Number of Cell = $(params.L * params.W * params.Height)")
    println()

    println("  @ Hamiltonian")
    println()
    if params.model == "Spin"
        println("    J0x = $(params.J0x)")
        println("    J0y = $(params.J0y)")
        println("    J0z = $(params.J0z)")
    else
        println("    t = $(params.t)")
        println("    U = $(params.U)")
    end
    println()

    println("  @ Numerical conditions")
    println()
    println("    NVMCCalMode = $(params.NVMCCalMode)")
    println("    NSROptItrStep = $(params.NSROptItrStep)")
    println("    NSROptItrSmp = $(params.NSROptItrSmp)")
    println("    NVMCSample = $(params.NVMCSample)")
    println("    2Sz = $(params.TwoSz)")
    println()
end
