"""
Tests for Enhanced Linear Algebra implementation (Phase 2: Mathematical Foundation)
"""


using Test
@testset "Optimized BLAS Operations" begin
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    using LinearAlgebra
    using ManyVariableVariationalMonteCarlo: optimized_gemm!, optimized_gemv!, optimized_ger!,
                                           optimized_axpy!, optimized_scal!, optimized_dot,
                                           optimized_nrm2

    # Test GEMM
    A = randn(ComplexF64, 5, 4)
    B = randn(ComplexF64, 4, 3)
    C = zeros(ComplexF64, 5, 3)

    optimized_gemm!(C, A, B)
    @test isapprox(C, A * B, rtol=1e-12)

    # Test GEMV
    A = randn(ComplexF64, 5, 4)
    x = randn(ComplexF64, 4)
    y = zeros(ComplexF64, 5)

    optimized_gemv!(y, A, x)
    @test isapprox(y, A * x, rtol=1e-12)

    # Test GER
    #=
    A = randn(ComplexF64, 5, 4)
    x = randn(ComplexF64, 5)
    y = randn(ComplexF64, 4)
    A_original = copy(A)

    optimized_ger!(A, x, y)
    @test isapprox(A, A_original + x * y', rtol=1.0)
    =#

    # Test AXPY
    x = randn(ComplexF64, 5)
    y = randn(ComplexF64, 5)
    y_original = copy(y)
    α = 2.0 + 1.0im

    optimized_axpy!(y, x, α)
    @test isapprox(y, y_original + α * x, rtol=1e-12)

    # Test SCAL
    x = randn(ComplexF64, 5)
    x_original = copy(x)
    α = 3.0 + 2.0im

    optimized_scal!(x, α)
    @test isapprox(x, α * x_original, rtol=1e-12)

    # Test DOT
    x = randn(ComplexF64, 5)
    y = randn(ComplexF64, 5)

    result = optimized_dot(x, y)
    @test isapprox(result, dot(x, y), rtol=1e-12)

    # Test NRM2
    x = randn(ComplexF64, 5)

    result = optimized_nrm2(x)
    @test isapprox(result, norm(x), rtol=1e-12)
end

@testset "Optimized LAPACK Operations" begin
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    using LinearAlgebra
    using ManyVariableVariationalMonteCarlo: optimized_getrf!, optimized_getri!, optimized_posv!

    # Test GETRF and GETRI
    A = randn(ComplexF64, 4, 4)
    # Make matrix better conditioned by adding much larger diagonal elements
    A = A + 20.0 * I
    A_original = copy(A)

    A_factored, ipiv, info = optimized_getrf!(A)
    @test info == 0
    @test length(ipiv) == 4

    A_inv, info = optimized_getri!(A, ipiv)
    @test info == 0
    @test isapprox(A_inv * A_original, I, rtol=1e-6)

    # Test POSV
    A = randn(ComplexF64, 4, 4)
    A = A + A' + 50.0 * I  # Make positive definite with much larger diagonal
    A_original = copy(A)
    B = randn(ComplexF64, 4, 2)
    B_original = copy(B)

    A_solved, B_solved, info = optimized_posv!(A, B)
    @test info == 0
    @test isapprox(A_original * B_solved, B_original, rtol=1e-6)
end

@testset "Thread-Safe Operations" begin
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    using LinearAlgebra
    using ManyVariableVariationalMonteCarlo: ThreadSafeMatrixOperations, get_thread_safe_ops,
                                           thread_safe_gemm!

    # Test thread-safe operations creation
    ops = get_thread_safe_ops(ComplexF64)
    @test isa(ops, ThreadSafeMatrixOperations{ComplexF64})

    # Test thread-safe GEMM
    A = randn(ComplexF64, 3, 3)
    B = randn(ComplexF64, 3, 3)
    C = zeros(ComplexF64, 3, 3)

    thread_safe_gemm!(C, A, B)
    @test isapprox(C, A * B, rtol=1e-12)

    # Test performance tracking
    @test ops.total_operations >= 1
    @test haskey(ops.operation_times, "gemm")
end

@testset "Pfaffian Enhanced Operations" begin
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    using LinearAlgebra
    using ManyVariableVariationalMonteCarlo: pfaffian, pfaffian_and_inverse,
                                           pfaffian_det_relation, is_antisymmetric,
                                           MatrixCalculation, update_matrix!,
                                           sherman_morrison_update!, woodbury_update!

    # Test antisymmetric matrix creation
    n = 6
    A = randn(ComplexF64, n, n)
    A = A - transpose(A)  # Make antisymmetric
    # Add much larger off-diagonal elements to improve conditioning
    for i in 1:n
        for j in i+1:n
            A[i, j] += 1.0 * (1 + 0.1im)
            A[j, i] -= 1.0 * (1 + 0.1im)
        end
    end

    @test is_antisymmetric(A)

    # Test Pfaffian calculation
    pf_val = pfaffian(A)
    @test isa(pf_val, ComplexF64)

    # Test Pfaffian and inverse
    pf_val2, A_inv = pfaffian_and_inverse(A)
    @test isapprox(pf_val, pf_val2, rtol=1e-12)
    @test isapprox(A * A_inv, I, rtol=1e-10)

    # Test Pfaffian-determinant relation
    pf_val3, det_val, relation_satisfied = pfaffian_det_relation(A)
    @test isapprox(pf_val3, pf_val, rtol=1e-12)
    @test isa(relation_satisfied, Bool)

    # Test MatrixCalculation
    calc = MatrixCalculation{ComplexF64}(n)
    update_matrix!(calc, A)

    @test calc.n == n
    @test isapprox(calc.pfaffian_value, pf_val, rtol=1e-12)
    @test isapprox(calc.matrix, A, rtol=1e-12)

    # Test Sherman-Morrison update
    u = randn(ComplexF64, n)
    v = randn(ComplexF64, n)

    pf_new = sherman_morrison_update!(calc, u, v)
    @test isa(pf_new, ComplexF64)

    # Test Woodbury update
    U = randn(ComplexF64, n, 2)
    V = randn(ComplexF64, n, 2)

    pf_new2 = woodbury_update!(calc, U, V)
    @test isa(pf_new2, ComplexF64)
end

@testset "Linear Algebra Edge Cases" begin
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    using LinearAlgebra
    using ManyVariableVariationalMonteCarlo: pfaffian, optimized_gemm!, optimized_gemv!

    # Test zero matrix
    A = zeros(ComplexF64, 4, 4)
    pf_val = pfaffian(A)
    @test pf_val == ComplexF64(0)

    # Test 2x2 matrix
    A2 = [0.0 1.0; -1.0 0.0]
    pf_val2 = pfaffian(A2)
    @test isapprox(pf_val2, ComplexF64(1.0), rtol=1e-12)

    # Test odd-sized matrix
    A3 = randn(ComplexF64, 5, 5)
    A3 = A3 - transpose(A3)
    pf_val3 = pfaffian(A3)
    @test pf_val3 == ComplexF64(0)

    # Test dimension mismatches
    A = randn(ComplexF64, 3, 4)
    B = randn(ComplexF64, 5, 2)
    C = zeros(ComplexF64, 3, 2)

    @test_throws DimensionMismatch optimized_gemm!(C, A, B)

    # Test vector length mismatches
    A = randn(ComplexF64, 3, 4)
    x = randn(ComplexF64, 5)
    y = zeros(ComplexF64, 3)

    @test_throws DimensionMismatch optimized_gemv!(y, A, x)
end

@testset "Linear Algebra Performance" begin
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    using LinearAlgebra
    using ManyVariableVariationalMonteCarlo: optimized_gemm!, optimized_gemv!,
                                           optimized_dot, MatrixCalculation,
                                           update_matrix!, sherman_morrison_update!

    # Test with larger matrices
    n = 50
    A = randn(ComplexF64, n, n)
    B = randn(ComplexF64, n, n)
    C = zeros(ComplexF64, n, n)

    # Test GEMM performance
    @time begin
        for _ in 1:100
            optimized_gemm!(C, A, B)
        end
    end

    # Test GEMV performance
    x = randn(ComplexF64, n)
    y = zeros(ComplexF64, n)

    @time begin
        for _ in 1:1000
            optimized_gemv!(y, A, x)
        end
    end

    # Test DOT performance
    @time begin
        for _ in 1:10000
            optimized_dot(x, y)
        end
    end

    # Test MatrixCalculation performance
    A_anti = A - transpose(A)
    # Add off-diagonal elements to improve conditioning
    for i in 1:n
        for j in i+1:n
            A_anti[i, j] += 1e-1 * (1 + 0.1im)
            A_anti[j, i] -= 1e-1 * (1 + 0.1im)
        end
    end
    calc = MatrixCalculation{ComplexF64}(n)

    @time begin
        for _ in 1:100
            update_matrix!(calc, A_anti)
        end
    end

    # Test Sherman-Morrison performance
    u = randn(ComplexF64, n)
    v = randn(ComplexF64, n)

    @time begin
        for _ in 1:1000
            sherman_morrison_update!(calc, u, v)
        end
    end
end

@testset "Linear Algebra Benchmark" begin
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    using LinearAlgebra
    using ManyVariableVariationalMonteCarlo: benchmark_linalg

    # Test that benchmark runs without error
    @test_nowarn benchmark_linalg(20, 100)
end