using Random
const AMP_MAX = 4.0

@inline _maxabs(buffer) = isempty(buffer) ? 0.0 : maximum(abs, buffer)

function _check_layout_mask(layout::ParameterLayout, mask::ParameterMask)
    length(mask.proj) == layout.nproj || throw(ArgumentError("proj mask length mismatch"))
    length(mask.rbm) == layout.nrbm || throw(ArgumentError("rbm mask length mismatch"))
    length(mask.slater) == layout.nslater ||
        throw(ArgumentError("slater mask length mismatch"))
    length(mask.opttrans) == layout.nopttrans ||
        throw(ArgumentError("opttrans mask length mismatch"))
    return nothing
end


function initialize_parameters!(
    params::ParameterSet,
    layout::ParameterLayout,
    mask::ParameterMask,
    flags::ParameterFlags;
    rng::AbstractRNG = Random.GLOBAL_RNG,
    para_qp_opttrans::AbstractVector{<:Number} = ComplexF64[],
    rbm_scale::Real = 1.0,
)
    _check_layout_mask(layout, mask)

    fill!(params.proj, 0)

    if flags.rbm_enabled
        for i in eachindex(params.rbm)
            if mask.rbm[i]
                if flags.all_complex
                    amp = 1e-2 * rand(rng)
                    angle = 2 * pi * rand(rng)
                    params.rbm[i] = amp * cis(angle)
                else
                    params.rbm[i] = 0.01 * (rand(rng) - 0.5) / max(rbm_scale, eps())
                end
            else
                params.rbm[i] = 0
            end
        end
    else
        fill!(params.rbm, 0)
    end

    if flags.all_complex
        for i in eachindex(params.slater)
            if mask.slater[i]
                real_part = 2 * (rand(rng) - 0.5)
                imag_part = 2 * (rand(rng) - 0.5)
                params.slater[i] = (real_part + imag_part * im) / sqrt(2)
            else
                params.slater[i] = 0
            end
        end
    else
        for i in eachindex(params.slater)
            params.slater[i] = mask.slater[i] ? 2 * (rand(rng) - 0.5) : 0
        end
    end

    if !isempty(params.opttrans)
        apply_opttrans_basis!(params, para_qp_opttrans)
    end

    amplitude = _maxabs(params.slater)
    if amplitude > AMP_MAX && amplitude > 0
        ratio = AMP_MAX / amplitude
        for i in eachindex(params.slater)
            params.slater[i] *= ratio
        end
    end

    return params
end

function apply_opttrans_basis!(params::ParameterSet, values::AbstractVector)
    ncopy = min(length(params.opttrans), length(values))
    for i = 1:ncopy
        params.opttrans[i] = values[i]
    end
    if ncopy < length(params.opttrans)
        fill!(view(params.opttrans, (ncopy+1):length(params.opttrans)), 0)
    end
    return params
end
