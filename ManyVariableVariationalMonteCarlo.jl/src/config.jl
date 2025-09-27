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
    width = facevalue(face, :W, Int; default=1)
    length = facevalue(face, :L, Int; default=1)
    width_sub = facevalue(face, :Wsub, Int; default=1)
    length_sub = facevalue(face, :Lsub, Int; default=1)
    nsublat = max(width_sub * length_sub, 1)
    nsites = width * length
    nsite_sub = max(div(nsites, nsublat), 1)
    model_sym = Symbol(facevalue(face, :model, String; default="UnknownModel"))
    lattice_sym = Symbol(facevalue(face, :lattice, String; default="UnknownLattice"))
    return SimulationConfig(face, String(root), nsublat, nsites, nsite_sub, model_sym, lattice_sym)
end

function load_face_definition(dir::AbstractString, filename::AbstractString)
    path = joinpath(dir, filename)
    face = load_face_definition(path)
    return face, SimulationConfig(face; root=dirname(path))
end
