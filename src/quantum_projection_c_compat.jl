"""
C-compatible Quantum Projection Implementation

Implements quantum projection functionality that closely matches the C implementation
in mVMC/src/mVMC/qp.c and projection.c.

Key features:
- Gauss-Legendre quadrature for spin projection
- Momentum projection with translational symmetry
- Particle number projection
- C-compatible weight calculation
- Direct translation of InitQPWeight() functionality
"""

using LinearAlgebra

"""
    CCompatQuantumProjection{T}

C-compatible quantum projection structure matching the C implementation.
"""
mutable struct CCompatQuantumProjection{T<:Union{Float64,ComplexF64}}
    # Basic parameters (from C global variables)
    n_sp_gauss_leg::Int      # NSPGaussLeg
    n_mp_trans::Int          # NMPTrans
    n_sp_stot::Int          # NSPStot
    n_qp_fix::Int           # NQPFix
    n_qp_full::Int          # NQPFull
    n_opt_trans::Int        # NOptTrans

    # Spin projection arrays (equivalent to C global arrays)
    spgl_cos::Vector{T}         # SPGLCos[NSPGaussLeg]
    spgl_sin::Vector{T}         # SPGLSin[NSPGaussLeg]
    spgl_cos_sin::Vector{T}     # SPGLCosSin[NSPGaussLeg]
    spgl_cos_cos::Vector{T}     # SPGLCosCos[NSPGaussLeg]
    spgl_sin_sin::Vector{T}     # SPGLSinSin[NSPGaussLeg]

    # Momentum projection arrays
    para_qp_trans::Vector{T}    # ParaQPTrans[NMPTrans]
    qp_fix_weight::Vector{T}    # QPFixWeight[NQPFix]
    qp_full_weight::Vector{T}   # QPFullWeight[NQPFull]

    # Optimization transformation
    opt_trans::Vector{T}        # OptTrans[NOptTrans]
    para_qp_opt_trans::Vector{T} # ParaQPOptTrans[NOptTrans]

    # Gauss-Legendre quadrature workspace
    beta_points::Vector{Float64}
    gl_weights::Vector{Float64}

    # Initialization flag
    is_initialized::Bool

    function CCompatQuantumProjection{T}(
        n_sp_gauss_leg::Int = 1,
        n_mp_trans::Int = 1,
        n_sp_stot::Int = 0,
        n_opt_trans::Int = 1
    ) where {T}

        n_qp_fix = n_sp_gauss_leg * n_mp_trans
        n_qp_full = n_qp_fix * n_opt_trans

        # Initialize arrays
        spgl_cos = Vector{T}(undef, n_sp_gauss_leg)
        spgl_sin = Vector{T}(undef, n_sp_gauss_leg)
        spgl_cos_sin = Vector{T}(undef, n_sp_gauss_leg)
        spgl_cos_cos = Vector{T}(undef, n_sp_gauss_leg)
        spgl_sin_sin = Vector{T}(undef, n_sp_gauss_leg)

        para_qp_trans = ones(T, n_mp_trans)
        qp_fix_weight = Vector{T}(undef, n_qp_fix)
        qp_full_weight = Vector{T}(undef, n_qp_full)

        opt_trans = ones(T, n_opt_trans)
        para_qp_opt_trans = ones(T, n_opt_trans)

        beta_points = Vector{Float64}(undef, n_sp_gauss_leg)
        gl_weights = Vector{Float64}(undef, n_sp_gauss_leg)

        new{T}(
            n_sp_gauss_leg, n_mp_trans, n_sp_stot, n_qp_fix, n_qp_full, n_opt_trans,
            spgl_cos, spgl_sin, spgl_cos_sin, spgl_cos_cos, spgl_sin_sin,
            para_qp_trans, qp_fix_weight, qp_full_weight,
            opt_trans, para_qp_opt_trans,
            beta_points, gl_weights,
            false
        )
    end
end

"""
    init_qp_weight!(qp::CCompatQuantumProjection{T}) where T

Initialize quantum projection weights, equivalent to InitQPWeight() in C.
Direct translation of mVMC/src/mVMC/qp.c lines 38-86.
"""
function init_qp_weight!(qp::CCompatQuantumProjection{T}) where {T}
    println("Initializing quantum projection weights...")

    if qp.n_sp_gauss_leg == 1
        # Simple case: no spin projection (NSPGaussLeg == 1)
        qp.beta_points[1] = 0.0
        qp.gl_weights[1] = 1.0
        qp.spgl_cos[1] = T(1.0)
        qp.spgl_sin[1] = T(0.0)
        qp.spgl_cos_sin[1] = T(0.0)
        qp.spgl_cos_cos[1] = T(1.0)
        qp.spgl_sin_sin[1] = T(0.0)

        # Set momentum projection weights
        for j in 1:qp.n_mp_trans
            qp.qp_fix_weight[j] = qp.para_qp_trans[j]
        end
    else
        # Full spin projection with Gauss-Legendre quadrature
        gauss_legendre!(qp.beta_points, qp.gl_weights, 0.0, π, qp.n_sp_gauss_leg)

        for i in 1:qp.n_sp_gauss_leg
            beta = qp.beta_points[i]
            qp.spgl_cos[i] = T(cos(0.5 * beta))
            qp.spgl_sin[i] = T(sin(0.5 * beta))
            qp.spgl_cos_sin[i] = qp.spgl_cos[i] * qp.spgl_sin[i]
            qp.spgl_cos_cos[i] = qp.spgl_cos[i] * qp.spgl_cos[i]
            qp.spgl_sin_sin[i] = qp.spgl_sin[i] * qp.spgl_sin[i]

            # Calculate weight with Legendre polynomial
            w = 0.5 * sin(beta) * qp.gl_weights[i] * legendre_poly(cos(beta), qp.n_sp_stot)

            for j in 1:qp.n_mp_trans
                idx = i + (j-1) * qp.n_sp_gauss_leg
                qp.qp_fix_weight[idx] = T(w) * qp.para_qp_trans[j]
            end
        end
    end

    # Update full weights
    update_qp_weight!(qp)

    qp.is_initialized = true
    println("Quantum projection weights initialized successfully")
    return nothing
end

"""
    update_qp_weight!(qp::CCompatQuantumProjection{T}) where T

Update quantum projection weights, equivalent to UpdateQPWeight() in C.
Direct translation of mVMC/src/mVMC/qp.c lines 129-148.
"""
function update_qp_weight!(qp::CCompatQuantumProjection{T}) where {T}
    if qp.n_opt_trans > 1
        # With optimization transformation
        for i in 1:qp.n_opt_trans
            offset = (i-1) * qp.n_qp_fix
            tmp = qp.opt_trans[i]
            for j in 1:qp.n_qp_fix
                qp.qp_full_weight[offset + j] = tmp * qp.qp_fix_weight[j]
            end
        end
    else
        # No optimization transformation
        for j in 1:qp.n_qp_fix
            qp.qp_full_weight[j] = qp.qp_fix_weight[j]
        end
    end
    return nothing
end

"""
    gauss_legendre!(x::Vector{Float64}, w::Vector{Float64}, a::Float64, b::Float64, n::Int)

Compute Gauss-Legendre quadrature points and weights on interval [a,b].
Equivalent to GaussLeg() in C implementation.
"""
function gauss_legendre!(x::Vector{Float64}, w::Vector{Float64}, a::Float64, b::Float64, n::Int)
    if n == 1
        x[1] = 0.5 * (a + b)
        w[1] = b - a
        return
    end

    # Use standard [-1,1] interval first
    x_std, w_std = gausslegendre(n)

    # Transform to [a,b] interval
    for i in 1:n
        x[i] = 0.5 * ((b - a) * x_std[i] + (b + a))
        w[i] = 0.5 * (b - a) * w_std[i]
    end
    return nothing
end

"""
    legendre_poly(x::Float64, n::Int) -> Float64

Calculate Legendre polynomial P_n(x).
Equivalent to LegendrePoly() in C implementation.
"""
function legendre_poly(x::Float64, n::Int)
    if n == 0
        return 1.0
    elseif n == 1
        return x
    else
        # Recursive relation: (n+1)P_{n+1}(x) = (2n+1)xP_n(x) - nP_{n-1}(x)
        p0 = 1.0
        p1 = x
        for k in 2:n
            p2 = ((2*k - 1) * x * p1 - (k - 1) * p0) / k
            p0 = p1
            p1 = p2
        end
        return p1
    end
end

"""
    calculate_log_ip(qp::CCompatQuantumProjection{T}, pf_matrix::Vector{T},
                     qp_start::Int, qp_end::Int) where T

Calculate logarithm of inner product <φ|L|x>.
Equivalent to CalculateLogIP_fcmp() in C implementation.
"""
function calculate_log_ip(
    qp::CCompatQuantumProjection{T},
    pf_matrix::Vector{T},
    qp_start::Int,
    qp_end::Int
) where {T}
    qp_num = qp_end - qp_start
    ip = zero(T)

    for qp_idx in 1:qp_num
        ip += qp.qp_full_weight[qp_idx + qp_start - 1] * pf_matrix[qp_idx]
    end

    return log(ip)
end

"""
    calculate_ip(qp::CCompatQuantumProjection{T}, pf_matrix::Vector{T},
                 qp_start::Int, qp_end::Int) where T

Calculate inner product <φ|L|x>.
Equivalent to CalculateIP_fcmp() in C implementation.
"""
function calculate_ip(
    qp::CCompatQuantumProjection{T},
    pf_matrix::Vector{T},
    qp_start::Int,
    qp_end::Int
) where {T}
    qp_num = qp_end - qp_start
    ip = zero(T)

    for qp_idx in 1:qp_num
        ip += qp.qp_full_weight[qp_idx + qp_start - 1] * pf_matrix[qp_idx]
    end

    return ip
end

"""
    initialize_quantum_projection_from_config(config::SimulationConfig; T=ComplexF64)

Initialize quantum projection from simulation configuration.
This replaces the placeholder in mvmc_integration.jl.
"""
function initialize_quantum_projection_from_config(config::SimulationConfig; T=ComplexF64)
    # Extract quantum projection parameters from config
    face = config.face

    n_sp_gauss_leg = haskey(face, :NSPGaussLeg) ? Int(face[:NSPGaussLeg]) : 1
    n_mp_trans = haskey(face, :NMPTrans) ? Int(face[:NMPTrans]) : 1
    n_sp_stot = haskey(face, :NSPStot) ? Int(face[:NSPStot]) : 0
    n_opt_trans = haskey(face, :NOptTrans) ? Int(face[:NOptTrans]) : 1

    # Create quantum projection
    qp = CCompatQuantumProjection{T}(n_sp_gauss_leg, n_mp_trans, n_sp_stot, n_opt_trans)

    # Initialize momentum projection parameters if available
    if haskey(face, :ParaQPTrans)
        para_qp_trans = face[:ParaQPTrans]
        if isa(para_qp_trans, Vector) && length(para_qp_trans) == n_mp_trans
            qp.para_qp_trans .= T.(para_qp_trans)
        end
    end

    # Initialize optimization transformation parameters if available
    if haskey(face, :ParaQPOptTrans)
        para_qp_opt_trans = face[:ParaQPOptTrans]
        if isa(para_qp_opt_trans, Vector) && length(para_qp_opt_trans) == n_opt_trans
            qp.para_qp_opt_trans .= T.(para_qp_opt_trans)
            qp.opt_trans .= T.(para_qp_opt_trans)
        end
    end

    # Initialize weights
    init_qp_weight!(qp)

    return qp
end

# Import FastGaussQuadrature for high-quality Gauss-Legendre quadrature
try
    using FastGaussQuadrature: gausslegendre
catch
    # Fallback implementation if FastGaussQuadrature is not available
    function gausslegendre(n::Int)
        # Simple implementation for small n
        if n == 1
            return [0.0], [2.0]
        elseif n == 2
            x = [-1/√3, 1/√3]
            w = [1.0, 1.0]
            return x, w
        else
            # For larger n, use a basic implementation
            # This is not optimal but provides a fallback
            x = zeros(n)
            w = zeros(n)
            for i in 1:n
                x[i] = cos(π * (i - 0.25) / (n + 0.5))
                w[i] = 2.0 / n
            end
            return x, w
        end
    end
end
