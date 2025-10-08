"""
Real-only Optimization Path

Optimized computation path for systems with real-valued variational parameters.
Corresponds to AllComplexFlag==0 in C implementation.

Based on mVMC C implementation:
- stcopt.c: Real-only stochastic optimization
- vmccal.c: Real-only calculations
"""

using LinearAlgebra
using Printf

"""
    RealOptimizationConfig

Configuration for real-only optimization.
"""
struct RealOptimizationConfig
    all_complex_flag::Bool    # false for real-only
    n_para::Int              # Number of parameters
    sr_opt_size::Int         # SR optimization matrix size
    dsr_opt_red_cut::Float64 # Diagonal reduction cutoff
    dsr_opt_sta_del::Float64 # Stabilization delta
    dsr_opt_step_dt::Float64 # Step size

    function RealOptimizationConfig(;
        all_complex_flag::Bool = false,
        n_para::Int,
        dsr_opt_red_cut::Float64 = 1e-10,
        dsr_opt_sta_del::Float64 = 1e-5,
        dsr_opt_step_dt::Float64 = 1e-2,
    )
        sr_opt_size = n_para + 1
        new(all_complex_flag, n_para, sr_opt_size,
            dsr_opt_red_cut, dsr_opt_sta_del, dsr_opt_step_dt)
    end
end

"""
    RealOptimizationState

State for real-only optimization including SR matrices.
"""
mutable struct RealOptimizationState
    config::RealOptimizationConfig

    # Real-valued arrays (when AllComplexFlag==0)
    sr_opt_oo_real::Vector{Float64}  # Overlap matrix (real)
    sr_opt_ho_real::Vector{Float64}  # Force vector (real)
    sr_opt_o_real::Vector{Float64}   # O derivatives (real)

    # Complex arrays (for compatibility)
    sr_opt_oo::Vector{ComplexF64}
    sr_opt_ho::Vector{ComplexF64}
    sr_opt_o::Vector{ComplexF64}

    # Parameters (real for spin systems)
    parameters::Vector{Float64}

    # Optimization history
    energy_history::Vector{Float64}
    parameter_history::Vector{Vector{Float64}}

    function RealOptimizationState(config::RealOptimizationConfig)
        n_para = config.n_para
        sr_opt_size = config.sr_opt_size

        # Initialize real arrays
        sr_opt_oo_real = zeros(Float64, sr_opt_size * sr_opt_size)
        sr_opt_ho_real = zeros(Float64, sr_opt_size)
        sr_opt_o_real = zeros(Float64, sr_opt_size)

        # Initialize complex arrays (for compatibility)
        sr_opt_oo = zeros(ComplexF64, 2 * sr_opt_size * (2 * sr_opt_size + 2))
        sr_opt_ho = zeros(ComplexF64, 2 * sr_opt_size)
        sr_opt_o = zeros(ComplexF64, 2 * sr_opt_size)

        # Initialize parameters with small random values
        parameters = randn(Float64, n_para) * 0.01

        new(
            config,
            sr_opt_oo_real,
            sr_opt_ho_real,
            sr_opt_o_real,
            sr_opt_oo,
            sr_opt_ho,
            sr_opt_o,
            parameters,
            Float64[],
            Vector{Float64}[],
        )
    end
end

"""
    convert_real_to_complex_sr_arrays!(state::RealOptimizationState)

Convert real SR arrays to complex format for compatibility.
Based on stcopt.c lines 67-81 in mVMC C implementation.

C実装参考: stcopt.c 1行目から192行目まで
"""
function convert_real_to_complex_sr_arrays!(state::RealOptimizationState)
    if !state.config.all_complex_flag
        sr_opt_size = state.config.sr_opt_size

        # Convert SROptOO_real to SROptOO
        for i in 0:(2*sr_opt_size*(2*sr_opt_size+2)-1)
            int_x = i % (2 * sr_opt_size)
            int_y = (i - int_x) ÷ (2 * sr_opt_size)

            if int_x % 2 == 0 && int_y % 2 == 0
                j = int_x ÷ 2 + (int_y ÷ 2) * sr_opt_size
                state.sr_opt_oo[i+1] = state.sr_opt_oo_real[j+1]
            else
                state.sr_opt_oo[i+1] = 0.0 + 0.0im
            end
        end

        # Convert SROptO_real to SROptO
        for i in 0:(sr_opt_size-1)
            state.sr_opt_o[2*i+1] = state.sr_opt_o_real[i+1]
            state.sr_opt_o[2*i+2] = 0.0
        end
    end

    return nothing
end

"""
    calculate_diagonal_elements_real!(state::RealOptimizationState) -> Vector{Float64}

Calculate diagonal elements of S matrix for real case.
Based on stcopt.c lines 83-94.
"""
function calculate_diagonal_elements_real!(state::RealOptimizationState)
    n_para = state.config.n_para
    sr_opt_size = state.config.sr_opt_size

    # Convert to complex format if needed
    convert_real_to_complex_sr_arrays!(state)

    # Calculate diagonal elements
    r = zeros(Float64, 2 * n_para)

    for pi in 0:(2*n_para-1)
        # r[pi] = S[pi+1][pi+1] = OO[pi+2][pi+2] - OO[0][pi+2]^2
        idx = (pi + 2) * (2 * sr_opt_size) + (pi + 2)
        if idx <= length(state.sr_opt_oo)
            r[pi+1] = real(state.sr_opt_oo[idx]) - real(state.sr_opt_oo[pi+3])^2
        end
    end

    return r
end

"""
    apply_diagonal_cutoff_real(
        r::Vector{Float64},
        config::RealOptimizationConfig
    ) -> (Vector{Int}, Int)

Apply diagonal cutoff threshold to determine which parameters to optimize.
Returns: (smat_to_para_idx, n_smat)
"""
function apply_diagonal_cutoff_real(
    r::Vector{Float64},
    config::RealOptimizationConfig,
)
    n_para = config.n_para

    # Find max diagonal element
    s_diag_max = maximum(r)

    # Calculate cutoff threshold
    diag_cut_threshold = s_diag_max * config.dsr_opt_red_cut

    # Determine which parameters to optimize
    smat_to_para_idx = Int[]
    cut_num = 0

    for pi in 1:(2*n_para)
        s_diag = r[pi]

        if s_diag < diag_cut_threshold
            cut_num += 1
        else
            push!(smat_to_para_idx, pi)
        end
    end

    n_smat = length(smat_to_para_idx)

    return (smat_to_para_idx, n_smat)
end

"""
    solve_sr_equations_real!(
        state::RealOptimizationState,
        smat_to_para_idx::Vector{Int},
        n_smat::Int
    ) -> Vector{Float64}

Solve SR equations for real-only case.
Returns parameter updates.
"""
function solve_sr_equations_real!(
    state::RealOptimizationState,
    smat_to_para_idx::Vector{Int},
    n_smat::Int,
)
    config = state.config
    sr_opt_size = config.sr_opt_size

    if n_smat == 0
        @warn "No parameters to optimize after diagonal cutoff"
        return zeros(Float64, config.n_para)
    end

    # Build reduced S matrix and force vector
    S_reduced = zeros(Float64, n_smat, n_smat)
    g_reduced = zeros(Float64, n_smat)

    for i in 1:n_smat
        pi = smat_to_para_idx[i]
        g_reduced[i] = real(state.sr_opt_ho[pi])

        for j in 1:n_smat
            pj = smat_to_para_idx[j]
            idx = (pi + 1) * (2 * sr_opt_size) + (pj + 1)
            if idx <= length(state.sr_opt_oo)
                S_reduced[i, j] = real(state.sr_opt_oo[idx])
            end
        end

        # Add stabilization
        S_reduced[i, i] += config.dsr_opt_sta_del
    end

    # Solve S * dx = -g
    dx_reduced = try
        -S_reduced \ g_reduced
    catch e
        @warn "Failed to solve SR equations: $e"
        zeros(Float64, n_smat)
    end

    # Map back to full parameter space
    dx = zeros(Float64, 2 * config.n_para)
    for (i, pi) in enumerate(smat_to_para_idx)
        dx[pi] = dx_reduced[i]
    end

    # Apply step size and extract real part only
    dx_para = zeros(Float64, config.n_para)
    for i in 1:config.n_para
        dx_para[i] = dx[2*i-1] * config.dsr_opt_step_dt
    end

    return dx_para
end

"""
    update_parameters_real!(
        state::RealOptimizationState,
        dx::Vector{Float64}
    )

Update parameters with real-valued updates.
"""
function update_parameters_real!(
    state::RealOptimizationState,
    dx::Vector{Float64},
)
    state.parameters .+= dx

    # Store in history
    push!(state.parameter_history, copy(state.parameters))

    return nothing
end

"""
    run_real_optimization_step!(
        state::RealOptimizationState,
        energy::Float64
    ) -> Bool

Run one step of real-only optimization.
Returns true if successful.
"""
function run_real_optimization_step!(
    state::RealOptimizationState,
    energy::Float64,
)
    # Calculate diagonal elements
    r = calculate_diagonal_elements_real!(state)

    # Apply diagonal cutoff
    (smat_to_para_idx, n_smat) = apply_diagonal_cutoff_real(r, state.config)

    if n_smat == 0
        return false
    end

    # Solve SR equations
    dx = solve_sr_equations_real!(state, smat_to_para_idx, n_smat)

    # Update parameters
    update_parameters_real!(state, dx)

    # Store energy
    push!(state.energy_history, energy)

    return true
end

"""
    detect_all_complex_flag(parameters::Vector{T}) where T

Detect if parameters are all real or contain complex values.
Returns false if all imaginary parts are zero.
"""
function detect_all_complex_flag(parameters::Vector{T}) where {T}
    if T <: Real
        return false
    elseif T <: Complex
        # Check if all imaginary parts are zero
        for param in parameters
            if abs(imag(param)) > 1e-15
                return true
            end
        end
        return false
    else
        return true
    end
end

"""
    create_real_optimization_state(
        params::StdFaceParameters,
        n_proj::Int,
        n_slater::Int,
        n_opttrans::Int
    ) -> RealOptimizationState

Create real optimization state from StdFace parameters.
"""
function create_real_optimization_state(
    params::StdFaceParameters,
    n_proj::Int,
    n_slater::Int,
    n_opttrans::Int,
)
    n_para = n_proj + n_slater + n_opttrans

    config = RealOptimizationConfig(
        all_complex_flag = false,  # Spin systems are real
        n_para = n_para,
        dsr_opt_red_cut = params.DSROptRedCut,
        dsr_opt_sta_del = params.DSROptStaDel,
        dsr_opt_step_dt = params.DSROptStepDt,
    )

    return RealOptimizationState(config)
end
