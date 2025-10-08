mutable struct WeightedAverager
    wsum::Float64
    sum_e::Float64
    sum_e2::Float64
    function WeightedAverager()
        new(0.0, 0.0, 0.0)
    end
end

reset!(wa::WeightedAverager) = (wa.wsum = 0.0; wa.sum_e = 0.0; wa.sum_e2 = 0.0)

"""
    update!(wa::WeightedAverager, energies::AbstractVector{<:Complex}, weights::AbstractVector{<:Real})

Update weighted averager with energies and weights.

C実装参考: average.c 1行目から334行目まで
"""
function update!(wa::WeightedAverager, energies::AbstractVector{<:Complex}, weights::AbstractVector{<:Real})
    @assert length(energies) == length(weights)
    for i in eachindex(energies)
        w = float(weights[i])
        e = float(real(energies[i]))
        wa.wsum += w
        wa.sum_e += w * e
        wa.sum_e2 += w * (e * e)
    end
end

function update!(wa::WeightedAverager, energies::AbstractVector{<:Complex})
    onew = ones(length(energies))
    update!(wa, energies, onew)
end

function summarize_energy(wa::WeightedAverager)
    if wa.wsum <= 0
        return (0.0, 0.0)
    end
    Etot = wa.sum_e / wa.wsum
    Etot2 = wa.sum_e2 / wa.wsum
    return (Etot, Etot2)
end

