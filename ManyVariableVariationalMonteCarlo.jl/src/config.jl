const _COMMENT_PATTERN = r"//"

function _strip_comment(line::AbstractString)
    idx = findfirst(_COMMENT_PATTERN, line)
    isnothing(idx) && return line
    return first(split(line, _COMMENT_PATTERN; limit=2))
end

function _parse_face_value(raw::AbstractString)
    value = strip(raw)
    if isempty(value)
        return nothing
    elseif startswith(value, '"') && endswith(value, '"')
        return value[2:end-1]
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
    return value
end

function load_face_definition(path::AbstractString)
    entries = Pair{Symbol,Any}[]
    for raw_line in eachline(path)
        line = strip(_strip_comment(raw_line))
        isempty(line) && continue
        if occursin('=', line)
            key_str, value_str = split(line, '='; limit=2)
            key = Symbol(strip(key_str))
            value = _parse_face_value(value_str)
            push!(entries, key => value)
        end
    end
    return FaceDefinition(entries)
end

function SimulationConfig(face::FaceDefinition; root::AbstractString = ".")
    # Lattice parameters
    width = facevalue(face, :W, Int; default=1)
    length = facevalue(face, :L, Int; default=1)
    width_sub = facevalue(face, :Wsub, Int; default=1)
    length_sub = facevalue(face, :Lsub, Int; default=1)
    nsublat = max(width_sub * length_sub, 1)
    nsites = width * length
    nsite_sub = max(div(nsites, nsublat), 1)

    # Model parameters
    model_sym = Symbol(facevalue(face, :model, String; default="UnknownModel"))
    lattice_sym = Symbol(facevalue(face, :lattice, String; default="UnknownLattice"))

    # Physical parameters
    t = facevalue(face, :t, Float64; default=1.0)
    u = facevalue(face, :U, Float64; default=0.0)
    nelec = facevalue(face, :nelec, Int; default=0)
    sz_total = facevalue(face, Symbol("2Sz"), Int; default=0)

    # VMC parameters
    nvmc_sample = facevalue(face, :NVMCSample, Int; default=1000)
    nsr_opt_itr_step = facevalue(face, :NSROptItrStep, Int; default=100)
    nsr_opt_itr_smp = facevalue(face, :NSROptItrSmp, Int; default=10)
    dsr_opt_red_cut = facevalue(face, :DSROptRedCut, Float64; default=1e-8)
    dsr_opt_sta_del = facevalue(face, :DSROptStaDel, Float64; default=1e-2)
    dsr_opt_step_dt = facevalue(face, :DSROptStepDt, Float64; default=1e-2)

    # Lanczos parameters
    nlanczos_mode = facevalue(face, :NLanczosMode, Int; default=0)
    nsp_gauss_leg = facevalue(face, :NSPGaussLeg, Int; default=1)

    # Calculation mode
    nvmc_cal_mode = facevalue(face, :NVMCCalMode, Int; default=0)

    return SimulationConfig(face, String(root), nsublat, nsites, nsite_sub, model_sym, lattice_sym,
                           t, u, nelec, sz_total, nvmc_sample, nsr_opt_itr_step, nsr_opt_itr_smp,
                           dsr_opt_red_cut, dsr_opt_sta_del, dsr_opt_step_dt, nlanczos_mode,
                           nsp_gauss_leg, nvmc_cal_mode)
end

function load_face_definition(dir::AbstractString, filename::AbstractString)
    path = joinpath(dir, filename)
    face = load_face_definition(path)
    return face, SimulationConfig(face; root=dirname(path))
end
