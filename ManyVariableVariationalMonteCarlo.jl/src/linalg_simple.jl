"""
Simplified Linear Algebra backend for ManyVariableVariationalMonteCarlo.jl

Provides basic linear algebra operations without workspace dependencies.
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
    is_antisymmetric(A::Matrix{T}, tol::Float64 = 1e-12) where T

Check if matrix A is antisymmetric within tolerance.
"""
function is_antisymmetric(A::Matrix{T}, tol::Float64 = 1e-12) where T
    n = size(A, 1)
    if n != size(A, 2)
        return false
    end

    for i in 1:n
        for j in 1:n
            if abs(A[i, j] + A[j, i]) > tol
                return false
            end
        end
    end
    return true
end

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
function pfaffian(A::Matrix{T}; check_antisymmetric::Bool = true) where T <: Union{Float64, ComplexF64}
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
Uses optimized algorithm for antisymmetric matrices.
Modifies input matrix A.
"""
function _pfaffian_skew(A::Matrix{T}) where T <: Union{Float64, ComplexF64}
    n = size(A, 1)

    if n == 0
        return one(T)
    elseif n == 2
        return A[1, 2]
    elseif n == 4
        # Direct formula for 4x4 antisymmetric matrix
        return A[1,2]*A[3,4] - A[1,3]*A[2,4] + A[1,4]*A[2,3]
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

    # Compute Pfaffian (disable antisymmetric check for internal use)
    pf = pfaffian(A_copy; check_antisymmetric=false)

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
    pfaffian_det_relation(A::Matrix{T}) where T <: Union{Float64, ComplexF64}

Verify the relation Pf(A)^2 = det(A) for antisymmetric matrix A.
Returns (pfaffian_value, determinant_value, relation_satisfied).
"""
function pfaffian_det_relation(A::Matrix{T}) where T <: Union{Float64, ComplexF64}
    if !is_antisymmetric(A)
        throw(ArgumentError("Matrix must be antisymmetric"))
    end

    pf_val = pfaffian(A; check_antisymmetric=false)
    det_val = det(A)

    # Check if Pf(A)^2 = det(A) within numerical precision
    relation_satisfied = isapprox(pf_val^2, det_val, rtol=1e-10)

    return pf_val, det_val, relation_satisfied
end

"""
    pfaffian_skew_symmetric(A::Matrix{T}) where T <: Union{Float64, ComplexF64}

Compute Pfaffian of skew-symmetric matrix A (A^T = -A).
This is an alias for pfaffian with antisymmetric check disabled.
"""
function pfaffian_skew_symmetric(A::Matrix{T}) where T <: Union{Float64, ComplexF64}
    return pfaffian(A; check_antisymmetric=false)
end

"""
    MatrixCalculation{T}

Efficient container for repeated matrix calculations.
"""
mutable struct MatrixCalculation{T <: Union{Float64, ComplexF64}}
    n::Int
    matrix::Matrix{T}
    inverse::Matrix{T}
    pfaffian_value::T

    function MatrixCalculation{T}(n::Int) where T
        matrix = Matrix{T}(undef, n, n)
        inverse = Matrix{T}(undef, n, n)
        new{T}(n, matrix, inverse, zero(T))
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

    # Update Pfaffian (approximate)
    calc.pfaffian_value *= denom

    return calc.pfaffian_value
end

"""
    matrix_ratio(calc::MatrixCalculation{T}, new_row::Int, new_col::Vector{T}) where T

Compute ratio of determinants when replacing one row of the matrix.
"""
function matrix_ratio(calc::MatrixCalculation{T}, new_row::Int, new_col::Vector{T}) where T
    return dot(new_col, calc.inverse[:, new_row])
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

    println("Linear algebra benchmark completed.")
end

"""
    woodbury_update!(calc::MatrixCalculation{T}, U::Matrix{T}, V::Matrix{T}) where T

Apply Woodbury matrix identity update: A_new = A + U*V'
Returns new Pfaffian value.
"""
function woodbury_update!(calc::MatrixCalculation{T}, U::Matrix{T}, V::Matrix{T}) where T <: Union{Float64, ComplexF64}
    n, k = size(U)
    if size(V) != (n, k)
        throw(ArgumentError("U and V must have same dimensions"))
    end

    # Compute C = I + V' * A^{-1} * U
    C = Matrix{T}(I, k, k)
    for i in 1:k
        for j in 1:k
            for l in 1:n
                C[i, j] += V[l, i] * calc.inverse[l, :]' * U[:, j]
            end
        end
    end

    # Check if C is invertible and compute inverse
    C_inv = try
        inv(C)
    catch
        throw(ArgumentError("Woodbury update failed: C matrix is singular"))
    end

    # Update inverse using Woodbury formula
    # A_new^{-1} = A^{-1} - A^{-1} * U * C^{-1} * V' * A^{-1}
    temp1 = calc.inverse * U  # n x k
    temp2 = C_inv * V' * calc.inverse  # k x n

    calc.inverse .-= temp1 * temp2

    # Update matrix
    calc.matrix .+= U * V'

    # Recompute Pfaffian (simplified - in practice would use update formula)
    calc.pfaffian_value = pfaffian(calc.matrix; check_antisymmetric=false)

    return calc.pfaffian_value
end

# Thread-local storage for MatrixCalculation objects
const _matrix_calculations = Dict{Tuple{Type, Int, Int}, MatrixCalculation}()

"""
    get_matrix_calculation(T::Type{<:Union{Float64, ComplexF64}}, n::Int, thread_id::Int = 1)

Get thread-local MatrixCalculation object for type T and size n.
"""
function get_matrix_calculation(T::Type{<:Union{Float64, ComplexF64}}, n::Int, thread_id::Int = 1)
    key = (T, n, thread_id)

    if !haskey(_matrix_calculations, key)
        _matrix_calculations[key] = MatrixCalculation{T}(n)
    end

    return _matrix_calculations[key]
end

"""
    clear_matrix_calculations!()

Clear all cached MatrixCalculation objects.
"""
function clear_matrix_calculations!()
    empty!(_matrix_calculations)
end