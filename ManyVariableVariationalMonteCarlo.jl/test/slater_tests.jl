@testset "SlaterMatrix basic functionality" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    using LinearAlgebra
    # Test SlaterMatrix creation
    slater = SlaterMatrix{ComplexF64}(2, 4)
    @test slater.n_elec == 2
    @test slater.n_orb == 4
    @test size(slater.matrix) == (2, 4)
    @test slater.det_value == 0.0
    @test slater.log_det_value == 0.0
    @test !slater.is_valid
end
@testset "SlaterDeterminant basic functionality" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    using LinearAlgebra
    # Test SlaterDeterminant creation
    slater = SlaterDeterminant{ComplexF64}(2, 4)
    @test slater.slater_matrix.n_elec == 2
    @test slater.slater_matrix.n_orb == 4
    @test size(slater.inverse_matrix) == (2, 2)
    @test length(slater.orbital_indices) == 2
    @test length(slater.orbital_signs) == 2
    @test slater.update_count == 0
    @test slater.last_update_row == -1
    @test slater.last_update_col == -1
end
@testset "SlaterDeterminant initialization" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    using LinearAlgebra
    # Test initialization with orbital matrix
    slater = SlaterDeterminant{ComplexF64}(2, 4)
    orbital_matrix = [1.0+0.0im 2.0+0.0im 3.0+0.0im 4.0+0.0im;
                      5.0+0.0im 6.0+0.0im 7.0+0.0im 8.0+0.0im]
    initialize_slater!(slater, orbital_matrix)
    @test slater.slater_matrix.is_valid
    @test slater.orbital_indices == [1, 2]
    @test slater.orbital_signs == [1, 1]
    @test slater.slater_matrix.matrix == orbital_matrix[1:2, 1:4]
end
@testset "SlaterDeterminant determinant computation" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    using LinearAlgebra
    # Test determinant computation
    slater = SlaterDeterminant{ComplexF64}(2, 2)
    orbital_matrix = [1.0+0.0im 2.0+0.0im;
                      3.0+0.0im 4.0+0.0im]
    initialize_slater!(slater, orbital_matrix)
    # Compute determinant manually
    expected_det = det(orbital_matrix[1:2, 1:2])
    @test isapprox(abs(slater.slater_matrix.det_value), abs(expected_det), rtol=1e-10)
    @test isapprox(slater.slater_matrix.log_det_value, log(abs(expected_det)), rtol=1e-10)
end
@testset "SlaterDeterminant single electron update" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    using LinearAlgebra
    # Test single electron update
    slater = SlaterDeterminant{ComplexF64}(2, 2)
    orbital_matrix = [1.0+0.0im 2.0+0.0im;
                      3.0+0.0im 4.0+0.0im]
    initialize_slater!(slater, orbital_matrix)
    original_det = slater.slater_matrix.det_value
    # Update element (1,1) from 1.0 to 2.0
    ratio = update_slater!(slater, 1, 1, 2.0+0.0im)
    @test slater.slater_matrix.matrix[1, 1] == 2.0+0.0im
    @test isapprox(slater.slater_matrix.det_value, original_det * ratio, rtol=1e-10)
    @test slater.update_count == 1
    @test slater.last_update_row == 1
    @test slater.last_update_col == 1
end
@testset "SlaterDeterminant two electron update" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    using LinearAlgebra
    # Test two electron update
    slater = SlaterDeterminant{ComplexF64}(2, 2)
    orbital_matrix = [1.0+0.0im 2.0+0.0im;
                      3.0+0.0im 4.0+0.0im]
    initialize_slater!(slater, orbital_matrix)
    original_det = slater.slater_matrix.det_value
    # Update elements (1,1) and (2,2)
    ratio = two_electron_update!(slater, 1, 1, 2.0+0.0im, 2, 2, 5.0+0.0im)
    @test slater.slater_matrix.matrix[1, 1] == 2.0+0.0im
    @test slater.slater_matrix.matrix[2, 2] == 5.0+0.0im
    @test isapprox(slater.slater_matrix.det_value, original_det * ratio, rtol=1e-10)
    @test slater.update_count == 1
    @test slater.last_update_row == -1  # Marked as two-electron update
    @test slater.last_update_col == -1
end
@testset "SlaterDeterminant utility functions" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    using StableRNGs
    using LinearAlgebra
    # Test utility functions
    slater = SlaterDeterminant{ComplexF64}(2, 2)
    orbital_matrix = [1.0+0.0im 2.0+0.0im;
                      3.0+0.0im 4.0+0.0im]
    initialize_slater!(slater, orbital_matrix)
    @test get_determinant_value(slater) == slater.slater_matrix.det_value
    @test get_log_determinant_value(slater) == slater.slater_matrix.log_det_value
    @test is_valid(slater)
    # Test reset
    reset_slater!(slater)
    @test !is_valid(slater)
    @test slater.slater_matrix.det_value == 0.0
    @test slater.slater_matrix.log_det_value == 0.0
    @test slater.update_count == 0
end
