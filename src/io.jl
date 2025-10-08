struct GreenFunctionEntry
    bra::NTuple{2,Int}
    ket::NTuple{2,Int}
    value::ComplexF64
end

struct GreenFunctionTable
    entries::Vector{GreenFunctionEntry}
end

Base.iterate(table::GreenFunctionTable) = iterate(table.entries)
Base.iterate(table::GreenFunctionTable, state) = iterate(table.entries, state)
Base.length(table::GreenFunctionTable) = length(table.entries)
Base.getindex(table::GreenFunctionTable, idx::Int) = table.entries[idx]

"""
    read_initial_green(path::AbstractString)

Read initial Green function data from file.

C実装参考: initfile.c 1行目から243行目まで
"""
function read_initial_green(path::AbstractString)
    entries = GreenFunctionEntry[]
    expected = 0
    for raw_line in eachline(path)
        line = strip(raw_line)
        isempty(line) && continue
        if startswith(line, "=") || startswith(line, "//")
            continue
        elseif startswith(line, "NCisAjs")
            parts = split(line)
            expected = parse(Int, parts[end])
            sizehint!(entries, expected)
            continue
        end
        parts = split(line)
        length(parts) < 6 && continue
        idx1 = parse(Int, parts[1])
        idx2 = parse(Int, parts[2])
        idx3 = parse(Int, parts[3])
        idx4 = parse(Int, parts[4])
        real_part = parse(Float64, parts[5])
        imag_part = parse(Float64, parts[6])
        value = ComplexF64(real_part, imag_part)
        push!(entries, GreenFunctionEntry((idx1, idx2), (idx3, idx4), value))
    end
    if expected != 0 && length(entries) != expected
        Base.@warn "Green function entry count mismatch" expected length(entries)
    end
    return GreenFunctionTable(entries)
end
