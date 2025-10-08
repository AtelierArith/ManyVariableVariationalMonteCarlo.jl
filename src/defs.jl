struct NamelistEntry
    key::Symbol
    path::String
end

struct Namelist
    entries::Vector{NamelistEntry}
end

Base.iterate(list::Namelist) = iterate(list.entries)
Base.iterate(list::Namelist, state) = iterate(list.entries, state)
Base.length(list::Namelist) = length(list.entries)
Base.firstindex(::Namelist) = 1
Base.lastindex(list::Namelist) = length(list)
Base.getindex(list::Namelist, idx::Int) = list.entries[idx]

"""
    load_namelist(path::AbstractString)

Load namelist from file.

C実装参考: readdef.c 1行目から2751行目まで
"""
function load_namelist(path::AbstractString)
    entries = NamelistEntry[]
    for raw in eachline(path)
        line = strip(raw)
        isempty(line) && continue
        startswith(line, "=") && continue
        parts = split(line)
        length(parts) < 2 && continue
        key = Symbol(parts[1])
        value = String(parts[2])
        push!(entries, NamelistEntry(key, value))
    end
    return Namelist(entries)
end

struct TransferEntry
    from_site::Int
    from_orbital::Int
    to_site::Int
    to_orbital::Int
    amplitude::ComplexF64
end

struct TransferTable
    entries::Vector{TransferEntry}
end

Base.iterate(table::TransferTable) = iterate(table.entries)
Base.iterate(table::TransferTable, state) = iterate(table.entries, state)
Base.length(table::TransferTable) = length(table.entries)
Base.firstindex(::TransferTable) = 1
Base.lastindex(table::TransferTable) = length(table)
Base.getindex(table::TransferTable, idx::Int) = table.entries[idx]

function read_transfer_table(path::AbstractString)
    entries = TransferEntry[]
    for raw in eachline(path)
        line = strip(raw)
        isempty(line) && continue
        startswith(line, "=") && continue
        first_char = first(line)
        !(first_char in ('0':'9')) && first_char != '-' && continue
        parts = split(line)
        length(parts) < 6 && continue
        i1 = parse(Int, parts[1])
        i2 = parse(Int, parts[2])
        j1 = parse(Int, parts[3])
        j2 = parse(Int, parts[4])
        real_part = parse(Float64, parts[5])
        imag_part = parse(Float64, parts[6])
        push!(entries, TransferEntry(i1, i2, j1, j2, ComplexF64(real_part, imag_part)))
    end
    return TransferTable(entries)
end

struct CoulombIntraEntry
    site::Int
    value::Float64
end

struct CoulombIntraTable
    entries::Vector{CoulombIntraEntry}
end

Base.iterate(table::CoulombIntraTable) = iterate(table.entries)
Base.iterate(table::CoulombIntraTable, state) = iterate(table.entries, state)
Base.length(table::CoulombIntraTable) = length(table.entries)
Base.firstindex(::CoulombIntraTable) = 1
Base.lastindex(table::CoulombIntraTable) = length(table)
Base.getindex(table::CoulombIntraTable, idx::Int) = table.entries[idx]

function read_coulomb_intra(path::AbstractString)
    entries = CoulombIntraEntry[]
    for raw in eachline(path)
        line = strip(raw)
        isempty(line) && continue
        startswith(line, "=") && continue
        first_char = first(line)
        !(first_char in ('0':'9')) && first_char != '-' && continue
        parts = split(line)
        length(parts) < 2 && continue
        idx = parse(Int, parts[1])
        value = parse(Float64, parts[2])
        push!(entries, CoulombIntraEntry(idx, value))
    end
    return CoulombIntraTable(entries)
end

struct InterAllEntry
    indices::NTuple{8,Int}
    value::ComplexF64
end

struct InterAllTable
    entries::Vector{InterAllEntry}
end

Base.iterate(table::InterAllTable) = iterate(table.entries)
Base.iterate(table::InterAllTable, state) = iterate(table.entries, state)
Base.length(table::InterAllTable) = length(table.entries)
Base.firstindex(::InterAllTable) = 1
Base.lastindex(table::InterAllTable) = length(table)
Base.getindex(table::InterAllTable, idx::Int) = table.entries[idx]

function read_interall_table(path::AbstractString)
    entries = InterAllEntry[]
    for raw in eachline(path)
        line = strip(raw)
        isempty(line) && continue
        startswith(line, "=") && continue
        first_char = first(line)
        !(first_char in ('0':'9')) && first_char != '-' && continue
        parts = split(line)
        length(parts) < 10 && continue
        idx_tuple = ntuple(i -> parse(Int, parts[i]), 8)
        real_part = parse(Float64, parts[9])
        imag_part = parse(Float64, parts[10])
        push!(entries, InterAllEntry(idx_tuple, ComplexF64(real_part, imag_part)))
    end
    return InterAllTable(entries)
end

struct GreenOneEntry
    bra::NTuple{2,Int}
    ket::NTuple{2,Int}
end

struct GreenOneTable
    entries::Vector{GreenOneEntry}
end

Base.iterate(table::GreenOneTable) = iterate(table.entries)
Base.iterate(table::GreenOneTable, state) = iterate(table.entries, state)
Base.length(table::GreenOneTable) = length(table.entries)
Base.firstindex(::GreenOneTable) = 1
Base.lastindex(table::GreenOneTable) = length(table)
Base.getindex(table::GreenOneTable, idx::Int) = table.entries[idx]

function read_greenone_indices(path::AbstractString)
    entries = GreenOneEntry[]
    for raw in eachline(path)
        line = strip(raw)
        isempty(line) && continue
        startswith(line, "=") && continue
        first_char = first(line)
        !(first_char in ('0':'9')) && first_char != '-' && continue
        parts = split(line)
        length(parts) < 4 && continue
        bra = (parse(Int, parts[1]), parse(Int, parts[2]))
        ket = (parse(Int, parts[3]), parse(Int, parts[4]))
        push!(entries, GreenOneEntry(bra, ket))
    end
    return GreenOneTable(entries)
end
