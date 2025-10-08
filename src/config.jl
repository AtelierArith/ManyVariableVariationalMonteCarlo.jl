const _COMMENT_PATTERN = r"//"

function _strip_comment(line::AbstractString)
    idx = findfirst(_COMMENT_PATTERN, line)
    isnothing(idx) && return line
    return first(split(line, _COMMENT_PATTERN; limit = 2))
end

function _parse_face_value(raw::AbstractString)
    value = strip(raw)
    if isempty(value)
        return nothing
    elseif startswith(value, '"') && endswith(value, '"')
        return value[2:(end-1)]
    elseif lowercase(value) == "true"
        return true
    elseif lowercase(value) == "false"
        return false
    end
    parsed_int = tryparse(Int, value)
    if parsed_int !== nothing
        return parsed_int
    end
    parsed_float = tryparse(Float64, value)
    if parsed_float !== nothing
        return parsed_float
    end
    parsed_complex = _try_parse_complex(value)
    if parsed_complex !== nothing
        return parsed_complex
    end
    return value
end

function _try_parse_complex(value::AbstractString)
    value = strip(value)

    if occursin('i', value) || occursin('I', value)
        value = replace(value, r"[iI]" => "")
        if occursin('+', value)
            parts = split(value, '+')
            if length(parts) == 2
                real_part = tryparse(Float64, strip(parts[1]))
                imag_part = tryparse(Float64, strip(parts[2]))
                if real_part !== nothing && imag_part !== nothing
                    return ComplexF64(real_part, imag_part)
                end
            end
        elseif occursin('-', value) && !startswith(value, '-')
            parts = rsplit(value, '-', limit = 2)
            if length(parts) == 2
                real_part = tryparse(Float64, strip(parts[1]))
                imag_part = tryparse(Float64, "-" * strip(parts[2]))
                if real_part !== nothing && imag_part !== nothing
                    return ComplexF64(real_part, imag_part)
                end
            end
        else
            imag_part = tryparse(Float64, value)
            if imag_part !== nothing
                return ComplexF64(0.0, imag_part)
            end
        end
    end

    if startswith(value, '(') && endswith(value, ')')
        inner = value[2:(end-1)]
        if occursin(',', inner)
            parts = split(inner, ',')
            if length(parts) == 2
                real_part = tryparse(Float64, strip(parts[1]))
                imag_part = tryparse(Float64, strip(parts[2]))
                if real_part !== nothing && imag_part !== nothing
                    return ComplexF64(real_part, imag_part)
                end
            end
        end
    end

    return nothing
end

struct VMCDefinitionFiles
    namelist::String
    modpara::Union{String,Nothing}
    locspn::Union{String,Nothing}
    trans::Union{String,Nothing}
    coulombintra::Union{String,Nothing}
    coulombinter::Union{String,Nothing}
    hund::Union{String,Nothing}
    pairhop::Union{String,Nothing}
    exchange::Union{String,Nothing}
    gutzwiller::Union{String,Nothing}
    jastrow::Union{String,Nothing}
    doublon2::Union{String,Nothing}
    doublon4::Union{String,Nothing}
    orbital::Union{String,Nothing}
    transopt::Union{String,Nothing}
    onebodyg::Union{String,Nothing}
    twobodyg::Union{String,Nothing}
    rbm::Union{String,Nothing}
    initial::Union{String,Nothing}
end

function VMCDefinitionFiles(namelist::String)
    VMCDefinitionFiles(
        namelist,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
    )
end

function load_face_definition(path::AbstractString)
    entries = Pair{Symbol,Any}[]
    for raw_line in eachline(path)
        line = strip(_strip_comment(raw_line))
        isempty(line) && continue

        # Handle both '=' format and space-separated format (for namelist.def)
        if occursin('=', line)
            key_str, value_str = split(line, '='; limit = 2)
            key = Symbol(strip(key_str))
            value = _parse_face_value(value_str)
            push!(entries, key => value)
        else
            # Handle space-separated format (namelist.def format)
            parts = split(line)
            if length(parts) >= 2
                key = Symbol(strip(parts[1]))
                value = strip(parts[2])
                push!(entries, key => value)
            end
        end
    end
    return FaceDefinition(entries)
end

function read_namelist_file(path::AbstractString)
    def_files = VMCDefinitionFiles(path)
    face = load_face_definition(path)

    # Extract file paths from face definition
    def_files = VMCDefinitionFiles(
        path,
        haskey(face, :ModPara) ? string(face[:ModPara]) : nothing,
        haskey(face, :LocSpin) ? string(face[:LocSpin]) : nothing,
        haskey(face, :Trans) ? string(face[:Trans]) : nothing,
        haskey(face, :CoulombIntra) ? string(face[:CoulombIntra]) : nothing,
        haskey(face, :CoulombInter) ? string(face[:CoulombInter]) : nothing,
        haskey(face, :Hund) ? string(face[:Hund]) : nothing,
        haskey(face, :PairHop) ? string(face[:PairHop]) : nothing,
        haskey(face, :Exchange) ? string(face[:Exchange]) : nothing,
        haskey(face, :Gutzwiller) ? string(face[:Gutzwiller]) : nothing,
        haskey(face, :Jastrow) ? string(face[:Jastrow]) : nothing,
        haskey(face, :DoublonHolon2site) ? string(face[:DoublonHolon2site]) : nothing,
        haskey(face, :DoublonHolon4site) ? string(face[:DoublonHolon4site]) : nothing,
        haskey(face, :Orbital) ? string(face[:Orbital]) : nothing,
        haskey(face, :TransOpt) ? string(face[:TransOpt]) : nothing,
        haskey(face, :OneBodyG) ? string(face[:OneBodyG]) : nothing,
        haskey(face, :TwoBodyG) ? string(face[:TwoBodyG]) : nothing,
        haskey(face, :RBM) ? string(face[:RBM]) : nothing,
        haskey(face, :Initial) ? string(face[:Initial]) : nothing,
    )

    return def_files, face
end

struct VMCParameters
    data_file_head::String
    para_file_head::String
    nvmc_cal_mode::Int
    nlanczos_mode::Int
    nstore_o::Int
    nsrcg::Int
    ndata_idx_start::Int
    ndata_qty_smp::Int
    nsite::Int
    ne::Int
    nup::Int
    nsize::Int
    nsite2::Int
    nz::Int
    two_sz::Int
    nsp_gauss_leg::Int
    nsp_stot::Int
    nmp_trans::Int
    nqp_full::Int
    nqp_fix::Int
    nsr_opt_itr_step::Int
    nsr_opt_itr_smp::Int
    nsr_opt_fix_smp::Int
    dsr_opt_red_cut::Float64
    dsr_opt_sta_del::Float64
    dsr_opt_step_dt::Float64
    nsr_opt_cg_max_iter::Int
    dsr_opt_cg_tol::Float64
    nvmc_warm_up::Int
    nvmc_interval::Int
    nvmc_sample::Int
    nex_update_path::Int
    nblock_update_size::Int
    rnd_seed::Int
    nsplit_size::Int
    ntotal_def_int::Int
    ntotal_def_double::Int
    nloc_spn::Int
    ntransfer::Int
    ncoulomb_intra::Int
    ncoulomb_inter::Int
    nhund::Int
    npair_hop::Int
    nexchange::Int
    ninter_all::Int
    ngutzwiller::Int
    njastrow::Int
    ndoublon_holon_2site::Int
    ndoublon_holon_4site::Int
    norbital_parallel::Int
    norbital_anti_parallel::Int
    norbital_general::Int
    ntrans_opt::Int
    nrbm_hidden::Int
    nrbm_visible::Int
    none_body_g::Int
    ntwo_body_g::Int
    ntwo_body_g_ex::Int
end

function VMCParameters()
    VMCParameters(
        "output",
        "para",
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1000,  # nsr_opt_itr_step (matches C implementation)
        100,   # nsr_opt_itr_smp (matches C implementation: NSROptItrStep/10)
        1,
        1e-3,  # dsr_opt_red_cut (matches C implementation)
        2e-2,  # dsr_opt_sta_del (matches C implementation)
        2e-2,  # dsr_opt_step_dt (matches C implementation)
        100,   # nsr_opt_cg_max_iter (matches C implementation)
        1e-10, # dsr_opt_cg_tol (matches C implementation)
        10,    # nvmc_warm_up (matches C implementation)
        1,
        10,    # nvmc_sample (matches C implementation default)
        0,
        1,
        11272,     # rnd_seed (matches C implementation)
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
end

function read_modpara_file(path::AbstractString)
    params = VMCParameters()
    # println("DEBUG: read_modpara_file: Reading from $path")
    for line in eachline(path)
        stripped = strip(_strip_comment(line))
        isempty(stripped) && continue

        # Support both '=' and whitespace as separators
        if occursin('=', stripped)
            key, value = split(stripped, '=', limit = 2)
            key = strip(key)
            value = strip(value)

            # println("DEBUG: read_modpara_file: Found key=$key, value=$value")
            params = _update_vmc_parameter(params, key, value)
        else
            # Try splitting by whitespace
            parts = split(stripped)
            if length(parts) >= 2
                key = parts[1]
                value = parts[2]

                # println("DEBUG: read_modpara_file: Found key=$key, value=$value (whitespace-separated)")
                params = _update_vmc_parameter(params, key, value)
            end
        end
    end
    # println("DEBUG: read_modpara_file: Final params.nsite=$(params.nsite), nsr_opt_itr_step=$(params.nsr_opt_itr_step)")
    return params
end

function _update_vmc_parameter(
    params::VMCParameters,
    key::AbstractString,
    value::AbstractString,
)
    parsed_value = _parse_face_value(value)

    # Create new VMCParameters with updated field
    fields = [getfield(params, field) for field in fieldnames(VMCParameters)]

    if key == "CDataFileHead"
        fields[1] = string(parsed_value)
    elseif key == "CParaFileHead"
        fields[2] = string(parsed_value)
    elseif key == "NVMCCalMode"
        fields[3] = Int(parsed_value)
    elseif key == "NLanczosMode"
        fields[4] = Int(parsed_value)
    elseif key == "NStoreO" || key == "NStore"
        fields[5] = Int(parsed_value)
    elseif key == "NSRCG"
        fields[6] = Int(parsed_value)
    elseif key == "NDataIdxStart"
        fields[7] = Int(parsed_value)
    elseif key == "NDataQtySmp"
        fields[8] = Int(parsed_value)
    elseif key == "Nsite"
        fields[9] = Int(parsed_value)
    elseif key == "Ne" || key == "Ncond"
        fields[10] = Int(parsed_value)
    elseif key == "Nup"
        fields[11] = Int(parsed_value)
    elseif key == "Nsize"
        fields[12] = Int(parsed_value)
    elseif key == "2Sz"
        fields[15] = Int(parsed_value)  # two_sz is field 15
    elseif key == "NSPGaussLeg"
        fields[16] = Int(parsed_value)  # nsp_gauss_leg is field 16
    elseif key == "NSPStot"
        fields[17] = Int(parsed_value)  # nsp_stot is field 17
    elseif key == "NMPTrans"
        fields[18] = Int(parsed_value)  # nmp_trans is field 18
    elseif key == "NSROptItrStep"
        fields[21] = Int(parsed_value)  # nsr_opt_itr_step is field 21
    elseif key == "NSROptItrSmp"
        fields[22] = Int(parsed_value)  # nsr_opt_itr_smp is field 22
    elseif key == "DSROptRedCut"
        fields[24] = Float64(parsed_value)  # dsr_opt_red_cut is field 24
    elseif key == "DSROptStaDel"
        fields[25] = Float64(parsed_value)  # dsr_opt_sta_del is field 25
    elseif key == "DSROptStepDt"
        fields[26] = Float64(parsed_value)  # dsr_opt_step_dt is field 26
    elseif key == "NVMCWarmUp"
        fields[29] = Int(parsed_value)  # nvmc_warm_up is field 29
    elseif key == "NVMCInterval"
        fields[30] = Int(parsed_value)  # nvmc_interval is field 30
    elseif key == "NVMCSample"
        fields[31] = Int(parsed_value)  # nvmc_sample is field 31
    elseif key == "NExUpdatePath"
        fields[32] = Int(parsed_value)  # nex_update_path is field 32
    elseif key == "RndSeed"
        fields[34] = Int(parsed_value)  # rnd_seed is field 34
    elseif key == "NSplitSize"
        fields[35] = Int(parsed_value)  # nsplit_size is field 35
    end

    return VMCParameters(fields...)
end

function SimulationConfig(face::FaceDefinition; root::AbstractString = ".")
    width = facevalue(face, :W, Int; default = 1)
    length = facevalue(face, :L, Int; default = 1)
    width_sub = facevalue(face, :Wsub, Int; default = 1)
    length_sub = facevalue(face, :Lsub, Int; default = 1)
    nsublat = max(width_sub * length_sub, 1)
    nsites = width * length
    nsite_sub = max(div(nsites, nsublat), 1)

    # Debug output
    # println("DEBUG: SimulationConfig creation: width=$width, length=$length, nsites=$nsites")

    model_sym = Symbol(facevalue(face, :model, String; default = "UnknownModel"))
    lattice_sym = Symbol(facevalue(face, :lattice, String; default = "UnknownLattice"))

    # Detect spin model from exchange interactions
    if model_sym == :UnknownModel
        # Check if we have exchange interactions (indicates spin model)
        if haskey(face, :Exchange) || haskey(face, :J)
            model_sym = :Spin
        end
    end

    t = facevalue(face, :t, Float64; default = 1.0)
    u = facevalue(face, :U, Float64; default = 0.0)
    nelec = facevalue(face, :nelec, Int; default = 0)
    sz_total = facevalue(face, Symbol("2Sz"), Int; default = 0)

    nvmc_sample = facevalue(face, :NVMCSample, Int; default = 1000)
    nsr_opt_itr_step = facevalue(face, :NSROptItrStep, Int; default = 100)
    nsr_opt_itr_smp = facevalue(face, :NSROptItrSmp, Int; default = 30)
    dsr_opt_red_cut = facevalue(face, :DSROptRedCut, Float64; default = 1e-3)
    dsr_opt_sta_del = facevalue(face, :DSROptStaDel, Float64; default = 2e-2)
    dsr_opt_step_dt = facevalue(face, :DSROptStepDt, Float64; default = 2e-2)  # Match C implementation default

    nlanczos_mode = facevalue(face, :NLanczosMode, Int; default = 0)
    nsp_gauss_leg = facevalue(face, :NSPGaussLeg, Int; default = 1)
    nvmc_cal_mode = facevalue(face, :NVMCCalMode, Int; default = 0)
    apbc = facevalue(face, :APFlag, Bool; default = false)
    twist_x = facevalue(face, :TwistX, Float64; default = 0.0)
    twist_y = facevalue(face, :TwistY, Float64; default = 0.0)
    flush_file = facevalue(face, :FlushFile, Bool; default = false)
    flush_interval = facevalue(face, :NFileFlushInterval, Int; default = 0)
    nvmc_warm_up =
        facevalue(face, :NVMCWarmUp, Int; default = max(50, div(nvmc_sample, 10)))
    nvmc_interval = facevalue(face, :NVMCInterval, Int; default = 1)

    # Additional parameters from C implementation
    n_data_qty_smp = facevalue(face, :NDataQtySmp, Int; default = 1)
    nex_update_path = facevalue(face, :NExUpdatePath, Int; default = 0)
    rnd_seed = facevalue(face, :RndSeed, Int; default = 11272)  # Match C default
    nsplit_size = facevalue(face, :NSplitSize, Int; default = 1)

    # SR optimization parameters
    nsr_opt_cg_max_iter = facevalue(face, :NSROptCGMaxIter, Int; default = 1000)
    dsr_opt_cg_tol = facevalue(face, :DSROptCGTol, Float64; default = 1e-6)

    # Complex/real flag
    all_complex_flag = facevalue(face, :AllComplexFlag, Bool; default = true)

    # RBM flag
    flag_rbm = facevalue(face, :FlagRBM, Bool; default = false)

    # Orbital flags
    iflg_orbital_general = facevalue(face, :iFlgOrbitalGeneral, Int; default = 0)

    # Fixed spin zone parameters
    two_sz = facevalue(face, :TwoSz, Int; default = sz_total)

    return SimulationConfig(
        face,
        String(root),
        nsublat,
        nsites,
        nsite_sub,
        model_sym,
        lattice_sym,
        t,
        u,
        nelec,
        sz_total,
        nvmc_sample,
        nsr_opt_itr_step,
        nsr_opt_itr_smp,
        dsr_opt_red_cut,
        dsr_opt_sta_del,
        dsr_opt_step_dt,
        nlanczos_mode,
        nsp_gauss_leg,
        nvmc_cal_mode,
        apbc,
        twist_x,
        twist_y,
        flush_file,
        flush_interval,
        nvmc_warm_up,
        nvmc_interval,
        n_data_qty_smp,
        nex_update_path,
        rnd_seed,
        nsplit_size,
        nsr_opt_cg_max_iter,
        dsr_opt_cg_tol,
        all_complex_flag,
        flag_rbm,
        iflg_orbital_general,
        two_sz,
    )
end

function load_face_definition(dir::AbstractString, filename::AbstractString)
    path = joinpath(dir, filename)
    face = load_face_definition(path)
    return face, SimulationConfig(face; root = dirname(path))
end

function read_transfer_file(path::AbstractString, nsite::Int)
    transfers = Tuple{Int,Int,Int,Int,ComplexF64}[]

    open(path, "r") do fp
        readline(fp)
        n_transfer = parse(Int, split(readline(fp))[2])

        for _ = 1:n_transfer
            line = readline(fp)
            parts = split(line)
            if length(parts) >= 5
                i = parse(Int, parts[1])
                j = parse(Int, parts[2])
                k = parse(Int, parts[3])
                l = parse(Int, parts[4])

                if length(parts) >= 6
                    real_part = parse(Float64, parts[5])
                    imag_part = parse(Float64, parts[6])
                    value = ComplexF64(real_part, imag_part)
                else
                    value = ComplexF64(parse(Float64, parts[5]), 0.0)
                end

                push!(transfers, (i, j, k, l, value))
            end
        end
    end

    return transfers
end

function read_coulomb_intra_file(path::AbstractString, nsite::Int)
    coulombs = Tuple{Int,Float64}[]

    open(path, "r") do fp
        readline(fp)
        n_coulomb = parse(Int, split(readline(fp))[2])

        for _ = 1:n_coulomb
            line = readline(fp)
            parts = split(line)
            if length(parts) >= 2
                i = parse(Int, parts[1])
                value = parse(Float64, parts[2])
                push!(coulombs, (i, value))
            end
        end
    end

    return coulombs
end

function read_locspn_file(path::AbstractString, nsite::Int)
    locspns = zeros(Int, nsite)

    open(path, "r") do fp
        readline(fp)
        n_locspn = parse(Int, split(readline(fp))[2])

        for _ = 1:n_locspn
            line = readline(fp)
            parts = split(line)
            if length(parts) >= 2
                i = parse(Int, parts[1]) + 1
                value = parse(Int, parts[2])
                if 1 <= i <= nsite
                    locspns[i] = value
                end
            end
        end
    end

    return locspns
end

function read_gutzwiller_file(path::AbstractString, nsite::Int)
    gutzwillers = Tuple{Int,ComplexF64,Bool}[]

    open(path, "r") do fp
        readline(fp)
        n_gutz = parse(Int, split(readline(fp))[2])
        complex_flag = parse(Int, split(readline(fp))[2])

        for _ = 1:n_gutz
            line = readline(fp)
            parts = split(line)
            if length(parts) >= 3
                i = parse(Int, parts[1])

                if complex_flag == 1 && length(parts) >= 4
                    real_part = parse(Float64, parts[2])
                    imag_part = parse(Float64, parts[3])
                    value = ComplexF64(real_part, imag_part)
                    opt_flag = parse(Int, parts[4]) == 1
                else
                    value = ComplexF64(parse(Float64, parts[2]), 0.0)
                    opt_flag = parse(Int, parts[3]) == 1
                end

                push!(gutzwillers, (i, value, opt_flag))
            end
        end
    end

    return gutzwillers
end

function load_vmc_configuration(namelist_path::AbstractString)
    def_files, face = read_namelist_file(namelist_path)

    params = if def_files.modpara !== nothing
        modpara_path = joinpath(dirname(namelist_path), def_files.modpara)
        read_modpara_file(modpara_path)
    else
        VMCParameters()
    end

    nsite = params.nsite
    ne = params.ne

    # println("DEBUG: load_vmc_configuration: nsite=$nsite, ne=$ne")

    fields = [getfield(params, field) for field in fieldnames(VMCParameters)]
    fields[9] = nsite
    fields[10] = ne
    fields[11] = ne
    fields[12] = 2 * ne
    fields[13] = 2 * nsite

    params = VMCParameters(fields...)

    # Add modpara.def parameters to face definition
    push_definition!(face, :NSROptItrStep, params.nsr_opt_itr_step)
    push_definition!(face, :NSROptItrSmp, params.nsr_opt_itr_smp)
    push_definition!(face, :DSROptRedCut, params.dsr_opt_red_cut)
    push_definition!(face, :DSROptStaDel, params.dsr_opt_sta_del)
    push_definition!(face, :DSROptStepDt, params.dsr_opt_step_dt)
    push_definition!(face, :NVMCWarmUp, params.nvmc_warm_up)
    push_definition!(face, :NVMCInterval, params.nvmc_interval)
    push_definition!(face, :NVMCSample, params.nvmc_sample)
    push_definition!(face, :RndSeed, params.rnd_seed)
    push_definition!(face, :NDataIdxStart, params.ndata_idx_start)
    push_definition!(face, :NDataQtySmp, params.ndata_qty_smp)
    push_definition!(face, :NExUpdatePath, params.nex_update_path)
    push_definition!(face, :NSplitSize, params.nsplit_size)
    push_definition!(face, :NSRCG, params.nsrcg)
    push_definition!(face, :NSPGaussLeg, params.nsp_gauss_leg)
    push_definition!(face, :NSPStot, params.nsp_stot)
    push_definition!(face, :NMPTrans, params.nmp_trans)
    push_definition!(face, :TwoSz, params.two_sz)

    # Add lattice parameters to face definition for SimulationConfig
    # For spin chain models, we need to set W and L based on nsite
    # println("DEBUG: load_vmc_configuration: About to add lattice parameters to face. nsite=$nsite")
    if nsite > 0
        # For 1D chain, assume L = nsite and W = 1
        push_definition!(face, :L, nsite)
        push_definition!(face, :W, 1)
        push_definition!(face, :Lsub, 1)
        push_definition!(face, :Wsub, 1)
        push_definition!(face, :Height, 1)
        push_definition!(face, :Hsub, 1)

        # Debug output
        # println("DEBUG: load_vmc_configuration: Added lattice parameters to face: L=$nsite, W=1")
        # println("DEBUG: load_vmc_configuration: face[:L] = $(face[:L])")
        # println("DEBUG: load_vmc_configuration: face[:W] = $(face[:W])")
    end

    # Create SimulationConfig with the updated face
    config = SimulationConfig(face; root = dirname(namelist_path))

    # println("DEBUG: Final config.nsr_opt_itr_step = $(config.nsr_opt_itr_step)")

    return def_files, params, face, config
end
