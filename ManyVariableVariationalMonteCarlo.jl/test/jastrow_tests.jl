"""
Tests for Jastrow factor implementation

Tests all Jastrow factor functionality including:
- Basic Jastrow factor calculations
- Gradient computations
- Parameter management
- Performance benchmarks
"""

using Test
using StableRNGs
using ManyVariableVariationalMonteCarlo

@testitem "Jastrow Factor Basic Functionality" begin
    using ManyVariableVariationalMonteCarlo

    # Test Jastrow factor creation
    jf = JastrowFactor{ComplexF64}(4, 2)
    @test jf.n_site == 4
    @test jf.n_elec == 2
    @test jf.n_spin == 2

    # Test parameter addition
    add_gutzwiller_parameter!(jf, 1, ComplexF64(0.1))
    add_density_density_parameter!(jf, 1, 2, ComplexF64(0.05))
    add_spin_spin_parameter!(jf, 1, 2, 1, 1, ComplexF64(0.02))
    add_three_body_parameter!(jf, 1, 2, 3, ComplexF64(0.01))

    @test length(jf.gutzwiller_params) == 1
    @test length(jf.density_density_params) == 1
    @test length(jf.spin_spin_params) == 1
    @test length(jf.three_body_params) == 1

    # Test parameter values
    @test jf.gutzwiller_params[1].value == ComplexF64(0.1)
    @test jf.density_density_params[1].value == ComplexF64(0.05)
    @test jf.spin_spin_params[1].value == ComplexF64(0.02)
    @test jf.three_body_params[1].value == ComplexF64(0.01)
end

@testitem "Jastrow Factor Calculations" begin
    using ManyVariableVariationalMonteCarlo

    # Create Jastrow factor
    jf = JastrowFactor{ComplexF64}(4, 2)

    # Add parameters
    add_gutzwiller_parameter!(jf, 1, ComplexF64(0.1))
    add_density_density_parameter!(jf, 1, 2, ComplexF64(0.05))

    # Test electron configuration
    ele_idx = [1, 2]
    ele_cfg = [1, 1, 0, 0]
    ele_num = [1, 1, 0, 0]

    # Test Jastrow factor calculation
    jastrow_val = jastrow_factor(jf, ele_idx, ele_cfg, ele_num)
    @test isa(jastrow_val, ComplexF64)
    @test jastrow_val != zero(ComplexF64)

    # Test log Jastrow factor
    log_jastrow_val = log_jastrow_factor(jf, ele_idx, ele_cfg, ele_num)
    @test isa(log_jastrow_val, ComplexF64)
    @test abs(exp(log_jastrow_val) - jastrow_val) < 1e-10

    # Test Jastrow ratio
    ele_idx_new = [1, 3]
    ele_cfg_new = [1, 0, 1, 0]
    ele_num_new = [1, 0, 1, 0]

    ratio = jastrow_ratio(jf, ele_idx, ele_cfg, ele_num, ele_idx_new, ele_cfg_new, ele_num_new)
    @test isa(ratio, ComplexF64)
    @test ratio != zero(ComplexF64)
end

@testitem "Jastrow Factor Gradients" begin
    using ManyVariableVariationalMonteCarlo

    # Create Jastrow factor
    jf = JastrowFactor{ComplexF64}(4, 2)

    # Add parameters
    add_gutzwiller_parameter!(jf, 1, ComplexF64(0.1))
    add_density_density_parameter!(jf, 1, 2, ComplexF64(0.05))
    add_spin_spin_parameter!(jf, 1, 2, 1, 1, ComplexF64(0.02))

    # Test electron configuration
    ele_idx = [1, 2]
    ele_cfg = [1, 1, 0, 0]
    ele_num = [1, 1, 0, 0]

    # Test gradient calculation
    gradient = jastrow_gradient(jf, ele_idx, ele_cfg, ele_num)
    @test isa(gradient, Vector{ComplexF64})
    @test length(gradient) == 3  # 1 gutzwiller + 1 density-density + 1 spin-spin

    # Test gradient values
    @test gradient[1] == ComplexF64(1.0)  # Gutzwiller gradient
    @test gradient[2] == ComplexF64(1.0)  # Density-density gradient
    @test gradient[3] == ComplexF64(1.0)  # Spin-spin gradient
end

@testitem "Jastrow Factor Parameter Management" begin
    using ManyVariableVariationalMonteCarlo

    # Create Jastrow factor
    jf = JastrowFactor{ComplexF64}(4, 2)

    # Add parameters
    add_gutzwiller_parameter!(jf, 1, ComplexF64(0.1))
    add_density_density_parameter!(jf, 1, 2, ComplexF64(0.05))

    # Test parameter count
    n_params = jastrow_parameter_count(jf)
    @test n_params == 2

    # Test parameter extraction
    params = get_jastrow_parameters(jf)
    @test length(params) == 2
    @test params[1] == ComplexF64(0.1)
    @test params[2] == ComplexF64(0.05)

    # Test parameter setting
    new_params = [ComplexF64(0.2), ComplexF64(0.1)]
    set_jastrow_parameters!(jf, new_params)

    updated_params = get_jastrow_parameters(jf)
    @test updated_params[1] == ComplexF64(0.2)
    @test updated_params[2] == ComplexF64(0.1)
end

@testitem "Jastrow Factor Performance" begin
    using ManyVariableVariationalMonteCarlo

    # Create Jastrow factor
    jf = JastrowFactor{ComplexF64}(10, 5)

    # Add many parameters
    for i in 1:10
        add_gutzwiller_parameter!(jf, i, ComplexF64(0.1))
    end

    for i in 1:9
        add_density_density_parameter!(jf, i, i+1, ComplexF64(0.05))
    end

    # Test electron configuration
    ele_idx = collect(1:5)
    ele_cfg = zeros(Int, 10)
    ele_cfg[1:5] .= 1
    ele_num = copy(ele_cfg)

    # Benchmark Jastrow factor calculation
    n_iterations = 1000
    @time begin
        for _ in 1:n_iterations
            jastrow_factor(jf, ele_idx, ele_cfg, ele_num)
        end
    end

    # Benchmark gradient calculation
    @time begin
        for _ in 1:n_iterations
            jastrow_gradient(jf, ele_idx, ele_cfg, ele_num)
        end
    end

    @test jf.total_evaluations == 2 * n_iterations
end

@testitem "Jastrow Factor Edge Cases" begin
    using ManyVariableVariationalMonteCarlo

    # Test empty Jastrow factor
    jf = JastrowFactor{ComplexF64}(4, 2)

    ele_idx = [1, 2]
    ele_cfg = [1, 1, 0, 0]
    ele_num = [1, 1, 0, 0]

    # Should return 1.0 for empty Jastrow factor
    jastrow_val = jastrow_factor(jf, ele_idx, ele_cfg, ele_num)
    @test jastrow_val == ComplexF64(1.0)

    log_jastrow_val = log_jastrow_factor(jf, ele_idx, ele_cfg, ele_num)
    @test log_jastrow_val == ComplexF64(0.0)

    # Test parameter bounds
    @test_throws ArgumentError add_gutzwiller_parameter!(jf, 0, ComplexF64(0.1))
    @test_throws ArgumentError add_gutzwiller_parameter!(jf, 5, ComplexF64(0.1))
    @test_throws ArgumentError add_density_density_parameter!(jf, 0, 1, ComplexF64(0.1))
    @test_throws ArgumentError add_density_density_parameter!(jf, 1, 5, ComplexF64(0.1))
end

@testitem "Jastrow Factor Complex vs Real" begin
    using ManyVariableVariationalMonteCarlo

    # Test complex Jastrow factor
    jf_complex = JastrowFactor{ComplexF64}(4, 2)
    add_gutzwiller_parameter!(jf_complex, 1, ComplexF64(0.1 + 0.2im))

    # Test real Jastrow factor
    jf_real = JastrowFactor{Float64}(4, 2)
    add_gutzwiller_parameter!(jf_real, 1, 0.1)

    ele_idx = [1, 2]
    ele_cfg = [1, 1, 0, 0]
    ele_num = [1, 1, 0, 0]

    # Test complex calculation
    jastrow_complex = jastrow_factor(jf_complex, ele_idx, ele_cfg, ele_num)
    @test isa(jastrow_complex, ComplexF64)
    @test imag(jastrow_complex) != 0.0

    # Test real calculation
    jastrow_real = jastrow_factor(jf_real, ele_idx, ele_cfg, ele_num)
    @test isa(jastrow_real, Float64)
    @test imag(jastrow_real) == 0.0
end
