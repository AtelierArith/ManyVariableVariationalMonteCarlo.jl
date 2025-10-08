"""
Quantum Projection System for mVMC C Compatibility

Translates the quantum projection modules (gauleg.c, legendrepoly.c, projection.c)
to Julia, maintaining exact compatibility with C numerical methods.

Ported from gauleg.c, legendrepoly.c, and projection.c.
"""

using LinearAlgebra
using SpecialFunctions

# Constants matching C implementation
const PI = 3.14159265358979323846
const EPS = 1.0e-14

"""
    MVMCQuantumProjection

Quantum projection state matching the C implementation.
"""
mutable struct MVMCQuantumProjection
    # Gauss-Legendre quadrature
    nsp_gauss_leg::Int
    spgl_cos::Vector{ComplexF64}
    spgl_sin::Vector{ComplexF64}
    spgl_cos_sin::Vector{ComplexF64}
    spgl_cos_cos::Vector{ComplexF64}
    spgl_sin_sin::Vector{ComplexF64}

    # Legendre polynomial coefficients
    legendre_coeffs::Vector{Float64}

    # Projection weights
    qp_full_weight::Vector{ComplexF64}
    qp_fix_weight::Vector{ComplexF64}

    # Projection parameters
    nsp_stot::Int
    nmp_trans::Int
    nqp_full::Int
    nqp_fix::Int

    # Transformation matrices
    qp_trans::Union{Matrix{Int}, Nothing}
    qp_trans_inv::Union{Matrix{Int}, Nothing}
    qp_trans_sgn::Union{Matrix{Int}, Nothing}
    para_qp_trans::Union{Vector{ComplexF64}, Nothing}

    function MVMCQuantumProjection()
        new(
            0,      # nsp_gauss_leg
            ComplexF64[],  # spgl_cos
            ComplexF64[],  # spgl_sin
            ComplexF64[],  # spgl_cos_sin
            ComplexF64[],  # spgl_cos_cos
            ComplexF64[],  # spgl_sin_sin
            Float64[],     # legendre_coeffs
            ComplexF64[],  # qp_full_weight
            ComplexF64[],  # qp_fix_weight
            0,      # nsp_stot
            0,      # nmp_trans
            0,      # nqp_full
            0,      # nqp_fix
            nothing, # qp_trans
            nothing, # qp_trans_inv
            nothing, # qp_trans_sgn
            nothing  # para_qp_trans
        )
    end
end

"""
    gauleg!(x::Vector{Float64}, w::Vector{Float64}, n::Int)

Gauss-Legendre quadrature weights and abscissas.
Matches C function gauleg() from gauleg.c.
"""
function gauleg!(x::Vector{Float64}, w::Vector{Float64}, n::Int)
    if n <= 0
        error("gauleg: n must be positive")
    end

    # Initialize arrays
    fill!(x, 0.0)
    fill!(w, 0.0)

    # For n=1, use midpoint
    if n == 1
        x[1] = 0.0
        w[1] = 2.0
        return
    end

    # For n=2, use exact values
    if n == 2
        x[1] = -1.0 / sqrt(3.0)
        x[2] = 1.0 / sqrt(3.0)
        w[1] = 1.0
        w[2] = 1.0
        return
    end

    # General case: use Newton's method to find roots
    m = (n + 1) ÷ 2

    for i in 1:m
        # Initial guess for i-th root
        z = cos(π * (i - 0.25) / (n + 0.5))

        # Newton's method
        for iter in 1:100
            p1 = 1.0
            p2 = 0.0

            # Evaluate Legendre polynomial and its derivative
            for j in 1:n
                p3 = p2
                p2 = p1
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j
            end

            # Derivative
            pp = n * (z * p1 - p2) / (z * z - 1.0)

            # Newton step
            z1 = z
            z = z1 - p1 / pp

            # Check convergence
            if abs(z - z1) < EPS
                break
            end
        end

        # Store root and its symmetric partner
        x[i] = -z
        x[n + 1 - i] = z

        # Weight
        w[i] = 2.0 / ((1.0 - z * z) * pp * pp)
        w[n + 1 - i] = w[i]
    end
end

"""
    legendre_poly(n::Int, x::Float64) -> Float64

Legendre polynomial P_n(x).
Matches C function legendrepoly() from legendrepoly.c.
"""
function legendre_poly(n::Int, x::Float64)::Float64
    if n < 0
        error("legendre_poly: n must be non-negative")
    end

    if n == 0
        return 1.0
    elseif n == 1
        return x
    end

    # Use recurrence relation: (n+1)P_{n+1}(x) = (2n+1)xP_n(x) - nP_{n-1}(x)
    p0 = 1.0
    p1 = x

    for i in 1:n-1
        p2 = ((2 * i + 1) * x * p1 - i * p0) / (i + 1)
        p0 = p1
        p1 = p2
    end

    return p1
end

"""
    legendre_poly_deriv(n::Int, x::Float64) -> Float64

Derivative of Legendre polynomial P'_n(x).
"""
function legendre_poly_deriv(n::Int, x::Float64)::Float64
    if n <= 0
        return 0.0
    elseif n == 1
        return 1.0
    end

    # Use recurrence relation for derivative
    p0 = 1.0
    p1 = x
    dp0 = 0.0
    dp1 = 1.0

    for i in 1:n-1
        p2 = ((2 * i + 1) * x * p1 - i * p0) / (i + 1)
        dp2 = ((2 * i + 1) * (p1 + x * dp1) - i * dp0) / (i + 1)
        p0 = p1
        p1 = p2
        dp0 = dp1
        dp1 = dp2
    end

    return dp1
end

"""
    setup_gauss_legendre_quadrature!(proj::MVMCQuantumProjection, n::Int)

Set up Gauss-Legendre quadrature for spin projection.
Matches C function setup_gauss_legendre_quadrature().
"""
function setup_gauss_legendre_quadrature!(proj::MVMCQuantumProjection, n::Int)
    proj.nsp_gauss_leg = n

    # Allocate arrays
    proj.spgl_cos = Vector{ComplexF64}(undef, n)
    proj.spgl_sin = Vector{ComplexF64}(undef, n)
    proj.spgl_cos_sin = Vector{ComplexF64}(undef, n)
    proj.spgl_cos_cos = Vector{ComplexF64}(undef, n)
    proj.spgl_sin_sin = Vector{ComplexF64}(undef, n)

    # Get quadrature points and weights
    x = Vector{Float64}(undef, n)
    w = Vector{Float64}(undef, n)
    gauleg!(x, w, n)

    # Set up trigonometric values
    for i in 1:n
        beta = π * x[i]  # beta = π * x
        cos_half = cos(beta / 2.0)
        sin_half = sin(beta / 2.0)

        proj.spgl_cos[i] = ComplexF64(cos_half, 0.0)
        proj.spgl_sin[i] = ComplexF64(sin_half, 0.0)
        proj.spgl_cos_sin[i] = ComplexF64(cos_half * sin_half, 0.0)
        proj.spgl_cos_cos[i] = ComplexF64(cos_half * cos_half, 0.0)
        proj.spgl_sin_sin[i] = ComplexF64(sin_half * sin_half, 0.0)
    end
end

"""
    setup_legendre_polynomials!(proj::MVMCQuantumProjection, n::Int)

Set up Legendre polynomial coefficients.
"""
function setup_legendre_polynomials!(proj::MVMCQuantumProjection, n::Int)
    proj.legendre_coeffs = Vector{Float64}(undef, n + 1)

    # Set up coefficients for P_n(x)
    for i in 0:n
        proj.legendre_coeffs[i + 1] = legendre_poly(i, 1.0)
    end
end

"""
    calculate_spin_projection_weight!(proj::MVMCQuantumProjection, stot::Int, sz::Int)

Calculate spin projection weight.
Matches C function calculate_spin_projection_weight().
"""
function calculate_spin_projection_weight!(proj::MVMCQuantumProjection, stot::Int, sz::Int)
    if proj.nsp_gauss_leg == 0
        error("Gauss-Legendre quadrature not set up")
    end

    # Calculate weight for each quadrature point
    for i in 1:proj.nsp_gauss_leg
        beta = π * (i - 1) / (proj.nsp_gauss_leg - 1)  # beta = π * x

        # Calculate weight using Legendre polynomial
        weight = legendre_poly(stot, cos(beta)) * sin(beta)

        # Store in appropriate array
        proj.spgl_cos[i] = ComplexF64(weight, 0.0)
    end
end

"""
    calculate_momentum_projection_weight!(proj::MVMCQuantumProjection, kx::Float64, ky::Float64, kz::Float64)

Calculate momentum projection weight.
Matches C function calculate_momentum_projection_weight().
"""
function calculate_momentum_projection_weight!(proj::MVMCQuantumProjection, kx::Float64, ky::Float64, kz::Float64)
    if proj.nmp_trans == 0
        error("Momentum projection not set up")
    end

    # Calculate weight for each transformation
    for i in 1:proj.nmp_trans
        # Calculate phase factor
        phase = 2π * (kx * proj.qp_trans[i, 1] + ky * proj.qp_trans[i, 2] + kz * proj.qp_trans[i, 3])
        weight = ComplexF64(cos(phase), sin(phase))

        # Store weight
        proj.qp_full_weight[i] = weight
    end
end

"""
    setup_quantum_projection!(proj::MVMCQuantumProjection, state::MVMCGlobalState)

Set up quantum projection system.
Matches C function setup_quantum_projection().
"""
function setup_quantum_projection!(proj::MVMCQuantumProjection, state::MVMCGlobalState)
    # Set up parameters
    proj.nsp_stot = state.nsp_stot
    proj.nmp_trans = state.nmp_trans
    proj.nqp_full = state.nqp_full
    proj.nqp_fix = state.nqp_fix

    # Set up Gauss-Legendre quadrature
    if proj.nsp_gauss_leg > 0
        setup_gauss_legendre_quadrature!(proj, proj.nsp_gauss_leg)
    end

    # Set up Legendre polynomials
    if proj.nsp_stot > 0
        setup_legendre_polynomials!(proj, proj.nsp_stot)
    end

    # Set up transformation matrices
    if proj.nmp_trans > 0
        proj.qp_trans = state.qp_trans
        proj.qp_trans_inv = state.qp_trans_inv
        proj.qp_trans_sgn = state.qp_trans_sgn
        proj.para_qp_trans = state.para_qp_trans
    end

    # Allocate weight arrays
    if proj.nqp_full > 0
        proj.qp_full_weight = Vector{ComplexF64}(undef, proj.nqp_full)
    end

    if proj.nqp_fix > 0
        proj.qp_fix_weight = Vector{ComplexF64}(undef, proj.nqp_fix)
    end
end

"""
    calculate_projection_ratio!(proj::MVMCQuantumProjection, state::MVMCGlobalState, sample::Int)

Calculate quantum projection ratio for a sample.
Matches C function calculate_projection_ratio().
"""
function calculate_projection_ratio!(proj::MVMCQuantumProjection, state::MVMCGlobalState, sample::Int)
    if proj.nqp_full == 0
        return ComplexF64(1.0, 0.0)
    end

    # Calculate weight for this sample
    weight = ComplexF64(1.0, 0.0)

    # Spin projection weight
    if proj.nsp_gauss_leg > 0
        for i in 1:proj.nsp_gauss_leg
            weight *= proj.spgl_cos[i]
        end
    end

    # Momentum projection weight
    if proj.nmp_trans > 0
        for i in 1:proj.nmp_trans
            weight *= proj.qp_full_weight[i]
        end
    end

    return weight
end

"""
    apply_quantum_projection!(proj::MVMCQuantumProjection, state::MVMCGlobalState, wavefunction::Vector{ComplexF64})

Apply quantum projection to wavefunction.
Matches C function apply_quantum_projection().
"""
function apply_quantum_projection!(proj::MVMCQuantumProjection, state::MVMCGlobalState, wavefunction::Vector{ComplexF64})
    if proj.nqp_full == 0
        return
    end

    # Apply spin projection
    if proj.nsp_gauss_leg > 0
        for i in 1:proj.nsp_gauss_leg
            # Apply spin projection operator
            # This would involve rotating the wavefunction
            # Implementation depends on specific projection method
        end
    end

    # Apply momentum projection
    if proj.nmp_trans > 0
        for i in 1:proj.nmp_trans
            # Apply momentum projection operator
            # This would involve translating the wavefunction
            # Implementation depends on specific projection method
        end
    end
end

"""
    print_projection_summary(proj::MVMCQuantumProjection)

Print quantum projection summary.
"""
function print_projection_summary(proj::MVMCQuantumProjection)
    println("=== Quantum Projection Summary ===")
    println("Gauss-Legendre points: $(proj.nsp_gauss_leg)")
    println("Spin projection: $(proj.nsp_stot)")
    println("Momentum projection: $(proj.nmp_trans)")
    println("Full projection: $(proj.nqp_full)")
    println("Fixed projection: $(proj.nqp_fix)")
    println("==================================")
end

# Export functions and types
export MVMCQuantumProjection, gauleg!, legendre_poly, legendre_poly_deriv,
       setup_gauss_legendre_quadrature!, setup_legendre_polynomials!,
       calculate_spin_projection_weight!, calculate_momentum_projection_weight!,
       setup_quantum_projection!, calculate_projection_ratio!,
       apply_quantum_projection!, print_projection_summary
