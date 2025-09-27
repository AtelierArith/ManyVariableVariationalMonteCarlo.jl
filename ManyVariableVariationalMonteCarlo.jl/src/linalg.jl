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
    pfaffian(A::Matrix{T}) where T <: Union{Float64, ComplexF64}

Compute Pfaffian of antisymmetric matrix A.
Uses optimized LAPACK-based implementation when available.
"""
function pfaffian(A::Matrix{T}) where T <: Union{Float64, ComplexF64}
    n = size(A, 1)
    if n != size(A, 2)
        throw(ArgumentError("Matrix must be square"))
    end
    if n % 2 != 0
        return zero(T)  # Pfaffian of odd-dimensional matrix is zero
    end

    # Check antisymmetry (optional, for debugging)
    # if !isapprox(A, -transpose(A))
    #     @warn "Matrix is not antisymmetric"
    # end

    return _pfaffian_skew(copy(A))
end

"""
    _pfaffian_skew(A::Matrix{T}) where T

Internal Pfaffian computation for antisymmetric matrix.
Modifies input matrix A.
"""
function _pfaffian_skew(A::Matrix{T}) where T <: Union{Float64, ComplexF64}
    n = size(A, 1)

    if n == 0
        return one(T)
    elseif n == 2
        return A[1, 2]
    end

    # Use pivot array
    ipiv = Vector{Int}(undef, n)

    # Perform LU decomposition with partial pivoting
    info = 0
    try
        if T <: Float64
            info = LAPACK.getrf!(A, ipiv)
        else  # ComplexF64
            info = LAPACK.getrf!(A, ipiv)
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
    for i in 1:n
        if ipiv[i] != i
            sign_pf *= -1
        end
    end

    # Compute product of diagonal elements (with appropriate signs)
    for i in 1:2:n-1
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
    pfaffian_and_inverse(A::Matrix{T}) where T

Compute both Pfaffian and inverse of antisymmetric matrix A.
More efficient than computing separately.
"""
function pfaffian_and_inverse(A::Matrix{T}) where T <: Union{Float64, ComplexF64}
    n = size(A, 1)
    A_copy = copy(A)

    # Compute Pfaffian
    pf = pfaffian(A_copy)

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
mutable struct MatrixCalculation{T <: Union{Float64, ComplexF64}}
    n::Int
    matrix::Matrix{T}
    inverse::Matrix{T}
    pfaffian_value::T
    workspace::Vector{T}
    int_workspace::Vector{Int}

    function MatrixCalculation{T}(n::Int) where T
        matrix = Matrix{T}(undef, n, n)
        inverse = Matrix{T}(undef, n, n)
        workspace = Vector{T}(undef, max(n*n, 1024))
        int_workspace = Vector{Int}(undef, n)
        new{T}(n, matrix, inverse, zero(T), workspace, int_workspace)
    end
end

"""
    update_matrix!(calc::MatrixCalculation{T}, new_matrix::Matrix{T}) where T

Update matrix and recompute Pfaffian and inverse.
"""
function update_matrix!(calc::MatrixCalculation{T}, new_matrix::Matrix{T}) where T
    copyto!(calc.matrix, new_matrix)
    calc.pfaffian_value, calc.inverse = pfaffian_and_inverse(calc.matrix)
    return calc.pfaffian_value
end

"""
    sherman_morrison_update!(calc::MatrixCalculation{T}, u::Vector{T}, v::Vector{T}) where T

Perform Sherman-Morrison rank-1 update of inverse matrix.
Updates inverse when matrix changes by A_new = A + u*v'.
"""
function sherman_morrison_update!(calc::MatrixCalculation{T}, u::Vector{T}, v::Vector{T}) where T
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
function woodbury_update!(calc::MatrixCalculation{T}, U::Matrix{T}, V::Matrix{T}) where T
    k = size(U, 2)
    if k != size(V, 2)
        throw(ArgumentError("U and V must have same number of columns"))
    end

    # Compute A^{-1} * U
    Ainv_U = calc.inverse * U

    # Compute (I + V' * A^{-1} * U)
    I_plus_VT_Ainv_U = I + transpose(V) * Ainv_U

    # Invert the k×k matrix
    try
        I_plus_VT_Ainv_U_inv = inv(I_plus_VT_Ainv_U)
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
function matrix_ratio(calc::MatrixCalculation{T}, new_row::Int, new_col::Vector{T}) where T
    # Ratio = det(A_new) / det(A_old)
    # When row i is replaced, ratio = new_col' * A^{-1}[:, i]

    return dot(new_col, calc.inverse[:, new_row])
end

"""
    optimal_lwork(::Type{T}, n::Int) where T

Determine optimal workspace size for LAPACK operations.
"""
function optimal_lwork(::Type{T}, n::Int) where T
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
    calculations::Vector{Union{MatrixCalculation{T}, Nothing}}

    function ThreadLocalMatrixCalculations{T}() where T
        new{T}([nothing for _ in 1:Threads.nthreads()])
    end
end

const THREAD_LOCAL_REAL_CALC = ThreadLocalMatrixCalculations{Float64}()
const THREAD_LOCAL_COMPLEX_CALC = ThreadLocalMatrixCalculations{ComplexF64}()

"""
    get_matrix_calculation(::Type{T}, n::Int, thread_id::Int = Threads.threadid()) where T

Get thread-local matrix calculation object, creating if necessary.
"""
function get_matrix_calculation(::Type{T}, n::Int, thread_id::Int = Threads.threadid()) where T
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
    benchmark_linalg(n::Int = 100, n_iterations::Int = 1000)

Benchmark linear algebra operations for VMC-typical usage.
"""
function benchmark_linalg(n::Int = 100, n_iterations::Int = 1000)
    println("Benchmarking linear algebra operations (n=$n, iterations=$n_iterations)...")

    # Generate test matrix
    A = randn(ComplexF64, n, n)
    A = A - transpose(A)  # Make antisymmetric

    # Benchmark Pfaffian calculation
    @time begin
        for _ in 1:n_iterations
            pfaffian(A)
        end
    end
    println("  Pfaffian calculation rate")

    # Benchmark matrix inversion
    @time begin
        for _ in 1:n_iterations
            inv(A)
        end
    end
    println("  Matrix inversion rate")

    # Benchmark combined Pfaffian and inverse
    @time begin
        for _ in 1:n_iterations
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
        for _ in 1:n_iterations÷10  # Fewer iterations since this modifies state
            try
                sherman_morrison_update!(calc, u, v)
                update_matrix!(calc, A)  # Reset for next iteration
            catch
                update_matrix!(calc, A)  # Reset on failure
            end
        end
    end
    println("  Sherman-Morrison update rate")

    println("Linear algebra benchmark completed.")
end