"""
Linear Algebra backend for ManyVariableVariationalMonteCarlo.jl

Provides optimized linear algebra operations including Pfaffian calculations,
matrix inversions, and specialized operations for VMC computations.
Ported from matrix.c in the C reference implementation.
"""

using LinearAlgebra
using LinearAlgebra.BLAS
using LinearAlgebra.LAPACK

"""
    PfaffianLimitError <: Exception

Exception thrown when Pfaffian value is below numerical threshold.
"""
struct PfaffianLimitError <: Exception
    value::Float64
    limit::Float64
end

Base.showerror(io::IO, e::PfaffianLimitError) =
    print(io, "PfaffianLimitError: Pfaffian value $(e.value) below limit $(e.limit)")

const PFAFFIAN_LIMIT = 1e-100

"""
    pfaffian(A::Matrix{T}; check_antisymmetric::Bool = true) where T <: Union{Float64, ComplexF64}

Compute Pfaffian of antisymmetric matrix A.
Uses optimized LAPACK-based implementation when available.

# Arguments
- `A`: Input matrix (should be antisymmetric)
- `check_antisymmetric`: Whether to verify that A is antisymmetric (default: true)

# Returns
- Pfaffian value of type T

# Throws
- `ArgumentError` if matrix is not square or not antisymmetric
- `PfaffianLimitError` if Pfaffian value is below numerical threshold
"""
function pfaffian(
    A::Matrix{T};
    check_antisymmetric::Bool = true,
) where {T<:Union{Float64,ComplexF64}}
    n = size(A, 1)
    if n != size(A, 2)
        throw(ArgumentError("Matrix must be square"))
    end
    if n % 2 != 0
        return zero(T)  # Pfaffian of odd-dimensional matrix is zero
    end

    if check_antisymmetric && !is_antisymmetric(A)
        throw(ArgumentError("Matrix must be antisymmetric for Pfaffian calculation"))
    end

    return _pfaffian_skew(copy(A))
end

"""
    _pfaffian_skew(A::Matrix{T}) where T

Internal Pfaffian computation for antisymmetric matrix.
Modifies input matrix A.
"""
function _pfaffian_skew(A::Matrix{T}) where {T<:Union{Float64,ComplexF64}}
    n = size(A, 1)

    if n == 0
        return one(T)
    elseif n == 2
        return A[1, 2]
    elseif n == 4
        # Direct formula for 4x4 matrix
        return A[1, 2] * A[3, 4] - A[1, 3] * A[2, 4] + A[1, 4] * A[2, 3]
    end

    # For larger matrices, use LU decomposition
    # Use pivot array
    ipiv = Vector{Int}(undef, n)

    # Perform LU decomposition with partial pivoting
    info = 0
    try
        if T <: Float64
            _, _, info = LAPACK.getrf!(A, ipiv)
        else  # ComplexF64
            _, _, info = LAPACK.getrf!(A, ipiv)
        end
    catch e
        if isa(e, LAPACKException)
            info = e.info
        else
            rethrow(e)
        end
    end

    if info > 0
        return zero(T)  # Matrix is singular
    end

    # Compute Pfaffian from LU factors
    pf = one(T)
    sign_pf = 1

    # Account for row permutations
    for i = 1:n
        if ipiv[i] != i
            sign_pf *= -1
        end
    end

    # Compute product of diagonal elements (with appropriate signs)
    for i = 1:2:(n-1)
        pf *= A[i, i+1]
    end

    result = sign_pf * pf

    # Check for numerical underflow
    if abs(result) < PFAFFIAN_LIMIT
        throw(PfaffianLimitError(abs(result), PFAFFIAN_LIMIT))
    end

    return result
end

"""
    pfaffian_det_relation(A::Matrix{T}) where T <: Union{Float64, ComplexF64}

Verify the relation Pf(A)^2 = det(A) for antisymmetric matrix A.
Returns (pfaffian_value, determinant_value, relation_satisfied).
"""
function pfaffian_det_relation(A::Matrix{T}) where {T<:Union{Float64,ComplexF64}}
    if !is_antisymmetric(A)
        throw(ArgumentError("Matrix must be antisymmetric"))
    end

    pf_val = pfaffian(A; check_antisymmetric = false)
    det_val = det(A)

    # Check if Pf(A)^2 = det(A) within numerical precision
    relation_satisfied = isapprox(pf_val^2, det_val, rtol = 1e-10)

    return pf_val, det_val, relation_satisfied
end

"""
    pfaffian_skew_symmetric(A::Matrix{T}) where T <: Union{Float64, ComplexF64}

Compute Pfaffian of skew-symmetric matrix A (A^T = -A).
This is an alias for pfaffian with antisymmetric check disabled.
"""
function pfaffian_skew_symmetric(A::Matrix{T}) where {T<:Union{Float64,ComplexF64}}
    return pfaffian(A; check_antisymmetric = false)
end

"""
    is_antisymmetric(A::Matrix{T}, tol::Float64 = 1e-12) where T

Check if matrix A is antisymmetric within tolerance.
"""
function is_antisymmetric(A::Matrix{T}, tol::Float64 = 1e-12) where {T}
    n = size(A, 1)
    if n != size(A, 2)
        return false
    end

    for i = 1:n
        for j = 1:n
            if abs(A[i, j] + A[j, i]) > tol
                return false
            end
        end
    end
    return true
end

"""
    pfaffian_and_inverse(A::Matrix{T}) where T

Compute both Pfaffian and inverse of antisymmetric matrix A.
More efficient than computing separately.
"""
function pfaffian_and_inverse(A::Matrix{T}) where {T<:Union{Float64,ComplexF64}}
    n = size(A, 1)
    A_copy = copy(A)

    # Compute Pfaffian (disable antisymmetric check for internal use)
    pf = pfaffian(A_copy; check_antisymmetric = false)

    # Compute inverse using LAPACK
    ipiv = Vector{Int}(undef, n)

    try
        if T <: Float64
            LAPACK.getrf!(A_copy, ipiv)
            LAPACK.getri!(A_copy, ipiv)
        else  # ComplexF64
            LAPACK.getrf!(A_copy, ipiv)
            LAPACK.getri!(A_copy, ipiv)
        end
    catch e
        if isa(e, LAPACKException)
            throw(ArgumentError("Matrix is singular and cannot be inverted"))
        else
            rethrow(e)
        end
    end

    return pf, A_copy
end

"""
    MatrixCalculation{T}

Efficient container for repeated matrix calculations with workspace reuse.
Stores matrices and intermediate results for VMC updates.
"""
mutable struct MatrixCalculation{T<:Union{Float64,ComplexF64}}
    n::Int
    matrix::Matrix{T}
    inverse::Matrix{T}
    pfaffian_value::T
    workspace::Vector{T}
    int_workspace::Vector{Int}

    function MatrixCalculation{T}(n::Int) where {T}
        matrix = Matrix{T}(undef, n, n)
        inverse = Matrix{T}(undef, n, n)
        workspace = Vector{T}(undef, max(n * n, 1024))
        int_workspace = Vector{Int}(undef, n)
        new{T}(n, matrix, inverse, zero(T), workspace, int_workspace)
    end
end

"""
    update_matrix!(calc::MatrixCalculation{T}, new_matrix::Matrix{T}) where T

Update matrix and recompute Pfaffian and inverse.
"""
function update_matrix!(calc::MatrixCalculation{T}, new_matrix::Matrix{T}) where {T}
    copyto!(calc.matrix, new_matrix)
    calc.pfaffian_value, calc.inverse = pfaffian_and_inverse(calc.matrix)
    return calc.pfaffian_value
end

"""
    sherman_morrison_update!(calc::MatrixCalculation{T}, u::Vector{T}, v::Vector{T}) where T

Perform Sherman-Morrison rank-1 update of inverse matrix.
Updates inverse when matrix changes by A_new = A + u*v'.
"""
function sherman_morrison_update!(
    calc::MatrixCalculation{T},
    u::Vector{T},
    v::Vector{T},
) where {T}
    # A^{-1} - (A^{-1}*u*v'*A^{-1}) / (1 + v'*A^{-1}*u)

    # Compute A^{-1} * u
    Ainv_u = calc.inverse * u

    # Compute v' * A^{-1}
    vT_Ainv = transpose(v) * calc.inverse

    # Compute denominator: 1 + v' * A^{-1} * u
    denom = one(T) + dot(v, Ainv_u)

    if abs(denom) < 1e-14
        throw(ArgumentError("Sherman-Morrison update failed: denominator too small"))
    end

    # Update inverse matrix
    calc.inverse .-= (Ainv_u * vT_Ainv) / denom

    # Update Pfaffian (if antisymmetric structure preserved)
    # This is approximate for general rank-1 updates
    calc.pfaffian_value *= denom

    return calc.pfaffian_value
end

"""
    woodbury_update!(calc::MatrixCalculation{T}, U::Matrix{T}, V::Matrix{T}) where T

Perform Woodbury matrix identity update for rank-k updates.
Updates inverse when matrix changes by A_new = A + U*V'.
"""
function woodbury_update!(calc::MatrixCalculation{T}, U::Matrix{T}, V::Matrix{T}) where {T}
    k = size(U, 2)
    if k != size(V, 2)
        throw(ArgumentError("U and V must have same number of columns"))
    end

    # Compute A^{-1} * U
    Ainv_U = calc.inverse * U

    # Compute (I + V' * A^{-1} * U)
    I_plus_VT_Ainv_U = I + transpose(V) * Ainv_U

    # Invert the k×k matrix
    I_plus_VT_Ainv_U_inv = try
        inv(I_plus_VT_Ainv_U)
    catch
        throw(ArgumentError("Woodbury update failed: inner matrix is singular"))
    end

    # Update inverse: A^{-1} - A^{-1}*U*(I + V'*A^{-1}*U)^{-1}*V'*A^{-1}
    calc.inverse .-= Ainv_U * I_plus_VT_Ainv_U_inv * transpose(V) * calc.inverse

    # Approximate Pfaffian update
    calc.pfaffian_value *= det(I_plus_VT_Ainv_U)

    return calc.pfaffian_value
end

"""
    matrix_ratio(calc::MatrixCalculation{T}, new_row::Int, new_col::Vector{T}) where T

Compute ratio of determinants when replacing one row of the matrix.
Useful for acceptance ratios in Monte Carlo updates.
"""
function matrix_ratio(
    calc::MatrixCalculation{T},
    new_row::Int,
    new_col::Vector{T},
) where {T}
    # Ratio = det(A_new) / det(A_old)
    # When row i is replaced, ratio = new_col' * A^{-1}[:, i]

    return dot(new_col, calc.inverse[:, new_row])
end

"""
    optimal_lwork(::Type{T}, n::Int) where T

Determine optimal workspace size for LAPACK operations.
"""
function optimal_lwork(::Type{T}, n::Int) where {T}
    # Query LAPACK for optimal workspace size
    A_dummy = Matrix{T}(undef, 1, 1)
    ipiv_dummy = Vector{Int}(undef, 1)

    if T <: Float64
        work_query = Vector{Float64}(undef, 1)
        LAPACK.getri!(A_dummy, ipiv_dummy, work_query, -1)
        return Int(work_query[1])
    else  # ComplexF64
        work_query = Vector{ComplexF64}(undef, 1)
        LAPACK.getri!(A_dummy, ipiv_dummy, work_query, -1)
        return Int(real(work_query[1]))
    end
end

"""
    ThreadLocalMatrixCalculations{T}

Thread-local storage for matrix calculations to avoid allocations.
"""
struct ThreadLocalMatrixCalculations{T}
    calculations::Vector{Union{MatrixCalculation{T},Nothing}}

    function ThreadLocalMatrixCalculations{T}() where {T}
        new{T}([nothing for _ = 1:Threads.nthreads()])
    end
end

const THREAD_LOCAL_REAL_CALC = ThreadLocalMatrixCalculations{Float64}()
const THREAD_LOCAL_COMPLEX_CALC = ThreadLocalMatrixCalculations{ComplexF64}()

"""
    get_matrix_calculation(::Type{T}, n::Int, thread_id::Int = Threads.threadid()) where T

Get thread-local matrix calculation object, creating if necessary.
"""
function get_matrix_calculation(
    ::Type{T},
    n::Int,
    thread_id::Int = Threads.threadid(),
) where {T}
    if T <: Float64
        storage = THREAD_LOCAL_REAL_CALC
    else
        storage = THREAD_LOCAL_COMPLEX_CALC
    end

    if thread_id > length(storage.calculations)
        throw(ArgumentError("Thread ID exceeds available storage"))
    end

    calc = storage.calculations[thread_id]
    if calc === nothing || calc.n != n
        calc = MatrixCalculation{T}(n)
        storage.calculations[thread_id] = calc
    end

    return calc
end

"""
    clear_matrix_calculations!()

Clear all thread-local matrix calculations.
"""
function clear_matrix_calculations!()
    # Clear real calculations
    for i = 1:length(THREAD_LOCAL_REAL_CALC.calculations)
        THREAD_LOCAL_REAL_CALC.calculations[i] = nothing
    end

    # Clear complex calculations
    for i = 1:length(THREAD_LOCAL_COMPLEX_CALC.calculations)
        THREAD_LOCAL_COMPLEX_CALC.calculations[i] = nothing
    end
end

"""
    optimized_gemm!(C::Matrix{T}, A::Matrix{T}, B::Matrix{T},
                   α::T = one(T), β::T = zero(T)) where T

Optimized matrix-matrix multiplication using BLAS.
"""
function optimized_gemm!(
    C::Matrix{T},
    A::Matrix{T},
    B::Matrix{T},
    α::T = one(T),
    β::T = zero(T),
) where {T<:Union{Float64,ComplexF64}}
    m, k = size(A)
    k2, n = size(B)

    if k != k2
        throw(DimensionMismatch("Inner dimensions must match"))
    end

    if size(C) != (m, n)
        throw(DimensionMismatch("Output matrix dimensions must match"))
    end

    # Use BLAS GEMM
    if T <: Float64
        BLAS.gemm!('N', 'N', α, A, B, β, C)
    else  # ComplexF64
        BLAS.gemm!('N', 'N', α, A, B, β, C)
    end

    return C
end

"""
    optimized_gemv!(y::Vector{T}, A::Matrix{T}, x::Vector{T},
                   α::T = one(T), β::T = zero(T)) where T

Optimized matrix-vector multiplication using BLAS.
"""
function optimized_gemv!(
    y::Vector{T},
    A::Matrix{T},
    x::Vector{T},
    α::T = one(T),
    β::T = zero(T),
) where {T<:Union{Float64,ComplexF64}}
    m, n = size(A)

    if length(x) != n
        throw(DimensionMismatch("Vector length must match matrix columns"))
    end

    if length(y) != m
        throw(DimensionMismatch("Output vector length must match matrix rows"))
    end

    # Use BLAS GEMV
    if T <: Float64
        BLAS.gemv!('N', α, A, x, β, y)
    else  # ComplexF64
        BLAS.gemv!('N', α, A, x, β, y)
    end

    return y
end

"""
    optimized_ger!(A::Matrix{T}, x::Vector{T}, y::Vector{T},
                  α::T = one(T)) where T

Optimized rank-1 update using BLAS.
"""
function optimized_ger!(
    A::Matrix{T},
    x::Vector{T},
    y::Vector{T},
    α::T = one(T),
) where {T<:Union{Float64,ComplexF64}}
    m, n = size(A)

    if length(x) != m
        throw(DimensionMismatch("Vector x length must match matrix rows"))
    end

    if length(y) != n
        throw(DimensionMismatch("Vector y length must match matrix columns"))
    end

    # Use BLAS GER
    if T <: Float64
        BLAS.ger!(α, x, y, A)
    else  # ComplexF64
        BLAS.geru!(α, x, y, A)
    end

    return A
end

"""
    optimized_axpy!(y::Vector{T}, x::Vector{T}, α::T = one(T)) where T

Optimized vector addition using BLAS.
"""
function optimized_axpy!(
    y::Vector{T},
    x::Vector{T},
    α::T = one(T),
) where {T<:Union{Float64,ComplexF64}}
    if length(x) != length(y)
        throw(DimensionMismatch("Vector lengths must match"))
    end

    # Use BLAS AXPY
    if T <: Float64
        BLAS.axpy!(α, x, y)
    else  # ComplexF64
        BLAS.axpy!(α, x, y)
    end

    return y
end

"""
    optimized_scal!(x::Vector{T}, α::T) where T

Optimized vector scaling using BLAS.
"""
function optimized_scal!(x::Vector{T}, α::T) where {T<:Union{Float64,ComplexF64}}
    # Use BLAS SCAL
    if T <: Float64
        BLAS.scal!(α, x)
    else  # ComplexF64
        BLAS.scal!(α, x)
    end

    return x
end

"""
    optimized_dot(x::Vector{T}, y::Vector{T}) where T

Optimized dot product using BLAS.
"""
function optimized_dot(x::Vector{T}, y::Vector{T}) where {T<:Union{Float64,ComplexF64}}
    if length(x) != length(y)
        throw(DimensionMismatch("Vector lengths must match"))
    end

    # Use BLAS DOT
    if T <: Float64
        return BLAS.dot(x, y)
    else  # ComplexF64
        return BLAS.dotc(x, y)
    end
end

"""
    optimized_nrm2(x::Vector{T}) where T

Optimized vector norm using BLAS.
"""
function optimized_nrm2(x::Vector{T}) where {T<:Union{Float64,ComplexF64}}
    # Use BLAS NRM2
    if T <: Float64
        return BLAS.nrm2(x)
    else  # ComplexF64
        return BLAS.nrm2(x)
    end
end

"""
    optimized_getrf!(A::Matrix{T}) where T

Optimized LU factorization using LAPACK.
"""
function optimized_getrf!(A::Matrix{T}) where {T<:Union{Float64,ComplexF64}}
    m, n = size(A)
    ipiv = Vector{Int}(undef, min(m, n))

    if T <: Float64
        _, _, info = LAPACK.getrf!(A, ipiv)
    else  # ComplexF64
        _, _, info = LAPACK.getrf!(A, ipiv)
    end

    return A, ipiv, info
end

"""
    optimized_getri!(A::Matrix{T}, ipiv::Vector{Int}) where T

Optimized matrix inversion using LAPACK.
"""
function optimized_getri!(
    A::Matrix{T},
    ipiv::Vector{Int},
) where {T<:Union{Float64,ComplexF64}}
    n = size(A, 1)

    if n != size(A, 2)
        throw(ArgumentError("Matrix must be square"))
    end

    if T <: Float64
        _, info = LAPACK.getri!(A, ipiv)
    else  # ComplexF64
        _, info = LAPACK.getri!(A, ipiv)
    end

    return A, Int(round(real(info)))
end

"""
    optimized_posv!(A::Matrix{T}, B::Matrix{T}) where T

Optimized positive definite system solve using LAPACK.
"""
function optimized_posv!(A::Matrix{T}, B::Matrix{T}) where {T<:Union{Float64,ComplexF64}}
    n = size(A, 1)

    if n != size(A, 2)
        throw(ArgumentError("Matrix A must be square"))
    end

    if size(B, 1) != n
        throw(DimensionMismatch("Matrix B rows must match A size"))
    end

    A_result, B_result = LAPACK.posv!('U', A, B)
    return A_result, B_result, 0  # LAPACK.posv! doesn't return info, assume success
end

"""
    ThreadSafeMatrixOperations{T}

Thread-safe wrapper for matrix operations with workspace management.
"""
mutable struct ThreadSafeMatrixOperations{T<:Union{Float64,ComplexF64}}
    # Thread-local workspaces
    gemm_workspace::Vector{Matrix{T}}
    gemv_workspace::Vector{Vector{T}}
    getrf_workspace::Vector{Vector{Int}}
    getri_workspace::Vector{Vector{T}}

    # Performance counters
    total_operations::Int
    operation_times::Dict{String,Float64}

    function ThreadSafeMatrixOperations{T}(max_threads::Int = Threads.nthreads()) where {T}
        gemm_workspace = [Matrix{T}(undef, 0, 0) for _ = 1:max_threads]
        gemv_workspace = [Vector{T}(undef, 0) for _ = 1:max_threads]
        getrf_workspace = [Vector{Int}(undef, 0) for _ = 1:max_threads]
        getri_workspace = [Vector{T}(undef, 0) for _ = 1:max_threads]

        operation_times = Dict{String,Float64}()

        new{T}(
            gemm_workspace,
            gemv_workspace,
            getrf_workspace,
            getri_workspace,
            0,
            operation_times,
        )
    end
end

const GLOBAL_MATRIX_OPS_REAL = ThreadSafeMatrixOperations{Float64}()
const GLOBAL_MATRIX_OPS_COMPLEX = ThreadSafeMatrixOperations{ComplexF64}()

"""
    get_thread_safe_ops(::Type{T}) where T

Get thread-safe matrix operations for type T.
"""
function get_thread_safe_ops(::Type{T}) where {T<:Union{Float64,ComplexF64}}
    if T <: Float64
        return GLOBAL_MATRIX_OPS_REAL
    else
        return GLOBAL_MATRIX_OPS_COMPLEX
    end
end

"""
    thread_safe_gemm!(C::Matrix{T}, A::Matrix{T}, B::Matrix{T},
                     α::T = one(T), β::T = zero(T)) where T

Thread-safe matrix-matrix multiplication.
"""
function thread_safe_gemm!(
    C::Matrix{T},
    A::Matrix{T},
    B::Matrix{T},
    α::T = one(T),
    β::T = zero(T),
) where {T<:Union{Float64,ComplexF64}}
    ops = get_thread_safe_ops(T)
    thread_id = Threads.threadid()

    # Ensure workspace is large enough
    m, n = size(C)
    if size(ops.gemm_workspace[thread_id]) != (m, n)
        ops.gemm_workspace[thread_id] = Matrix{T}(undef, m, n)
    end

    # Perform operation
    start_time = time()
    optimized_gemm!(C, A, B, α, β)
    end_time = time()

    # Update statistics
    ops.total_operations += 1
    ops.operation_times["gemm"] =
        get(ops.operation_times, "gemm", 0.0) + (end_time - start_time)

    return C
end

"""
    benchmark_linalg(n::Int = 100, n_iterations::Int = 1000)

Benchmark linear algebra operations for VMC-typical usage.
"""
function benchmark_linalg(n::Int = 100, n_iterations::Int = 1000)
    println("Benchmarking linear algebra operations (n=$n, iterations=$n_iterations)...")

    # Generate test matrix with better conditioning
    A = randn(ComplexF64, n, n)
    A = A - transpose(A)  # Make antisymmetric
    # Add small off-diagonal elements to improve conditioning while maintaining antisymmetry
    for i = 1:n
        for j = (i+1):n
            A[i, j] += 1e-2 * (1 + 0.1im)
            A[j, i] -= 1e-2 * (1 + 0.1im)
        end
    end

    # Benchmark Pfaffian calculation
    @time begin
        for _ = 1:n_iterations
            pfaffian(A)
        end
    end
    println("  Pfaffian calculation rate")

    # Benchmark matrix inversion
    @time begin
        for _ = 1:n_iterations
            inv(A)
        end
    end
    println("  Matrix inversion rate")

    # Benchmark combined Pfaffian and inverse
    @time begin
        for _ = 1:n_iterations
            pfaffian_and_inverse(A)
        end
    end
    println("  Combined Pfaffian + inverse rate")

    # Benchmark Sherman-Morrison update
    calc = MatrixCalculation{ComplexF64}(n)
    update_matrix!(calc, A)
    u = randn(ComplexF64, n)
    v = randn(ComplexF64, n)

    @time begin
        for _ = 1:(n_iterations÷10)  # Fewer iterations since this modifies state
            try
                sherman_morrison_update!(calc, u, v)
                update_matrix!(calc, A)  # Reset for next iteration
            catch
                update_matrix!(calc, A)  # Reset on failure
            end
        end
    end
    println("  Sherman-Morrison update rate")

    # Benchmark optimized BLAS operations
    B = randn(ComplexF64, n, n)
    C = zeros(ComplexF64, n, n)
    x = randn(ComplexF64, n)
    y = zeros(ComplexF64, n)

    @time begin
        for _ = 1:n_iterations
            optimized_gemm!(C, A, B)
        end
    end
    println("  Optimized GEMM rate")

    @time begin
        for _ = 1:n_iterations
            optimized_gemv!(y, A, x)
        end
    end
    println("  Optimized GEMV rate")

    @time begin
        for _ = 1:n_iterations
            optimized_dot(x, y)
        end
    end
    println("  Optimized DOT rate")

    println("Linear algebra benchmark completed.")
end
