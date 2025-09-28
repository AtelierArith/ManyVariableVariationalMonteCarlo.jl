"""
    FaceDefinition(entries)

Structured representation of `StdFace.def` style configuration files.
`entries` is a `Vector` of key/value pairs stored in the order they
appear in the definition file so downstream code can reproduce the same
ordering when writing files back to disk.
"""
struct FaceDefinition
    entries::Vector{Pair{Symbol,Any}}
end

FaceDefinition() = FaceDefinition(Pair{Symbol,Any}[])

function Base.getindex(def::FaceDefinition, key::Symbol)
    for (k, v) in def.entries
        k === key && return v
    end
    throw(KeyError(key))
end

function Base.haskey(def::FaceDefinition, key::Symbol)
    for (k, _) in def.entries
        k === key && return true
    end
    return false
end

function push_definition!(def::FaceDefinition, key::Symbol, value)
    push!(def.entries, key => value)
    return def
end

"""
    facevalue(def, key, ::Type{T}; default=nothing)

Typed lookup helper that attempts to convert the stored value to `T`.
When `default` is provided it will be returned when `key` is absent;
otherwise a `KeyError` is thrown.
"""
function facevalue(def::FaceDefinition, key::Symbol, ::Type{T}; default = nothing) where {T}
    if !haskey(def, key)
        default === nothing && throw(KeyError(key))
        return default
    end
    value = def[key]
    return value isa T ? value : convert(T, value)
end

Base.iterate(def::FaceDefinition) = iterate(def.entries)
Base.iterate(def::FaceDefinition, state) = iterate(def.entries, state)
Base.length(def::FaceDefinition) = length(def.entries)
Base.isempty(def::FaceDefinition) = isempty(def.entries)

"""
Immutable container for basic simulation metadata extracted from
`StdFace.def` and derived values such as the total number of lattice
sites.
"""
struct SimulationConfig
    face::FaceDefinition
    root::String
    nsublat::Int
    nsites::Int
    nsite_sub::Int
    model::Symbol
    lattice::Symbol
    # Physical parameters
    t::Float64
    u::Float64
    nelec::Int
    sz_total::Int
    # VMC parameters
    nvmc_sample::Int
    nsr_opt_itr_step::Int
    nsr_opt_itr_smp::Int
    dsr_opt_red_cut::Float64
    dsr_opt_sta_del::Float64
    dsr_opt_step_dt::Float64
    # Lanczos parameters
    nlanczos_mode::Int
    nsp_gauss_leg::Int
    # Calculation mode
    nvmc_cal_mode::Int
    # Boundary conditions
    apbc::Bool
    # Optional twist angles (radians) for boundary wraps in x/y
    twist_x::Float64
    twist_y::Float64
    # Output flush control
    flush_file::Bool
    flush_interval::Int
    # Sampling control
    nvmc_warm_up::Int
    nvmc_interval::Int
end

"""
Book-keeping describing how many variational parameters are present for
each family in the current simulation setup.
"""
struct ParameterLayout
    nproj::Int
    nrbm::Int
    nslater::Int
    nopttrans::Int
end

Base.length(layout::ParameterLayout) =
    layout.nproj + layout.nrbm + layout.nslater + layout.nopttrans

"""
Control flags that affect how the variational parameter buffers are
initialised.
"""
struct ParameterFlags
    all_complex::Bool
    rbm_enabled::Bool
end

"""
Logical masks that mark which degrees of freedom are optimised.
"""
struct ParameterMask
    proj::BitVector
    rbm::BitVector
    slater::BitVector
    opttrans::BitVector
end

ParameterMask(layout::ParameterLayout; default::Bool = false) = ParameterMask(
    BitVector(fill(default, layout.nproj)),
    BitVector(fill(default, layout.nrbm)),
    BitVector(fill(default, layout.nslater)),
    BitVector(fill(default, layout.nopttrans)),
)

"""
    ParameterSet(layout; T=ComplexF64)

Allocate variational parameter buffers with the supplied element type `T`
(defaulting to `ComplexF64`).
"""
mutable struct ParameterSet{
    Vp<:AbstractVector{<:Number},
    Vr<:AbstractVector{<:Number},
    Vs<:AbstractVector{<:Number},
    Vo<:AbstractVector{<:Number},
}
    proj::Vp
    rbm::Vr
    slater::Vs
    opttrans::Vo
end

function ParameterSet(layout::ParameterLayout; T::Type = ComplexF64)
    zero_vec(n) = Vector{T}(undef, n)
    return ParameterSet(
        zero_vec(layout.nproj),
        zero_vec(layout.nrbm),
        zero_vec(layout.nslater),
        zero_vec(layout.nopttrans),
    )
end

function Base.copy(params::ParameterSet)
    return ParameterSet(
        copy(params.proj),
        copy(params.rbm),
        copy(params.slater),
        copy(params.opttrans),
    )
end

Base.length(params::ParameterSet) =
    length(params.proj) +
    length(params.rbm) +
    length(params.slater) +
    length(params.opttrans)

function Base.show(io::IO, params::ParameterSet)
    print(
        io,
        "ParameterSet(proj=$(length(params.proj)), rbm=$(length(params.rbm)), slater=$(length(params.slater)), opttrans=$(length(params.opttrans)))",
    )
end

"""
Returns the maximum amplitude across all variational parameters. This is
useful when matching the amplitude clamp implemented in the C reference
(`D_AmpMax`).
"""
function maxabs(params::ParameterSet)
    maxval = 0.0
    for buffer in (params.proj, params.rbm, params.slater, params.opttrans)
        isempty(buffer) && continue
        local_max = maximum(abs, buffer)
        maxval = max(maxval, local_max)
    end
    return maxval
end
