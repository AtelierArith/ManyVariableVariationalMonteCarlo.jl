# Quantum Projection Integration Tests
# Tests for quantum projection initialization and integration

@testitem "initialize_quantum_projection_from_config basic functionality" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    # Create configuration with quantum projection parameters
    face = FaceDefinition()
    push_definition!(face, :NSPGaussLeg, 8)
    push_definition!(face, :NMPTrans, 1)
    push_definition!(face, :NSPStot, 0)
    push_definition!(face, :NOptTrans, 1)

    config = SimulationConfig(face)

    # Test initialization
    qp = initialize_quantum_projection_from_config(config; T=ComplexF64)

    @test qp isa ManyVariableVariationalMonteCarlo.CCompatQuantumProjection{ComplexF64}
    @test qp.n_sp_gauss_leg == 8
    @test qp.n_mp_trans == 1
    @test qp.n_sp_stot == 0
    @test qp.n_opt_trans == 1
    @test qp.is_initialized == true
end

@testitem "initialize_quantum_projection_from_config with default values" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    # Create minimal configuration
    face = FaceDefinition()
    push_definition!(face, :L, 4)
    push_definition!(face, :model, "Spin")

    config = SimulationConfig(face)

    # Test initialization with defaults
    qp = initialize_quantum_projection_from_config(config; T=ComplexF64)

    @test qp isa ManyVariableVariationalMonteCarlo.CCompatQuantumProjection{ComplexF64}
    @test qp.n_sp_gauss_leg == 1  # Default
    @test qp.n_mp_trans == 1      # Default
    @test qp.n_sp_stot == 0       # Default
    @test qp.n_opt_trans == 1     # Default
end

@testitem "initialize_quantum_projection_from_config with custom parameters" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    # Create configuration with custom parameters
    face = FaceDefinition()
    push_definition!(face, :NSPGaussLeg, 16)
    push_definition!(face, :NMPTrans, 4)
    push_definition!(face, :NSPStot, 2)
    push_definition!(face, :NOptTrans, 2)
    push_definition!(face, :ParaQPTrans, [1.0, 0.5, -0.5, -1.0])
    push_definition!(face, :ParaQPOptTrans, [1.0 + 0.0im, 0.0 + 1.0im])

    config = SimulationConfig(face)

    # Test initialization
    qp = initialize_quantum_projection_from_config(config; T=ComplexF64)

    @test qp.n_sp_gauss_leg == 16
    @test qp.n_mp_trans == 4
    @test qp.n_sp_stot == 2
    @test qp.n_opt_trans == 2

    # Test parameter arrays
    @test length(qp.para_qp_trans) == 4
    @test qp.para_qp_trans ≈ [1.0, 0.5, -0.5, -1.0]

    @test length(qp.para_qp_opt_trans) == 2
    @test qp.para_qp_opt_trans[1] ≈ 1.0 + 0.0im
    @test qp.para_qp_opt_trans[2] ≈ 0.0 + 1.0im
end

@testitem "print_quantum_projection_summary enabled case" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    # Create enhanced simulation with quantum projection
    face = FaceDefinition()
    push_definition!(face, :L, 4)
    push_definition!(face, :model, "Spin")
    push_definition!(face, :NSPGaussLeg, 8)
    push_definition!(face, :NMPTrans, 2)
    push_definition!(face, :NSPStot, 1)

    config = SimulationConfig(face)
    layout = ParameterLayout(2, 0, 0, 0)

    sim = ManyVariableVariationalMonteCarlo.EnhancedVMCSimulation{ComplexF64}(config, layout)
    sim.quantum_projection = initialize_quantum_projection_from_config(config; T=ComplexF64)

    # Capture output
    output_str = let
        temp_file = tempname()
        open(temp_file, "w") do io
            redirect_stdout(io) do
                print_quantum_projection_summary(sim)
            end
        end
        result = read(temp_file, String)
        rm(temp_file, force=true)
        result
    end

    # Check output content
    @test contains(output_str, "Quantum Projection Summary")
    @test contains(output_str, "ENABLED")
    @test contains(output_str, "Spin projection points: 8")
    @test contains(output_str, "Momentum projections: 2")
    @test contains(output_str, "Total spin: 1")
    @test contains(output_str, "Gauss-Legendre quadrature")
end

@testitem "print_quantum_projection_summary disabled case" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    # Create enhanced simulation without quantum projection
    face = FaceDefinition()
    push_definition!(face, :L, 4)
    push_definition!(face, :model, "Spin")

    config = SimulationConfig(face)
    layout = ParameterLayout(2, 0, 0, 0)

    sim = ManyVariableVariationalMonteCarlo.EnhancedVMCSimulation{ComplexF64}(config, layout)
    sim.quantum_projection = nothing

    # Capture output
    output_str = let
        temp_file = tempname()
        open(temp_file, "w") do io
            redirect_stdout(io) do
                print_quantum_projection_summary(sim)
            end
        end
        result = read(temp_file, String)
        rm(temp_file, force=true)
        result
    end

    # Check output content
    @test contains(output_str, "Quantum Projection Summary")
    @test contains(output_str, "DISABLED")
    @test contains(output_str, "Reason:")
end

@testitem "get_quantum_projection_info function" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    # Test with enabled quantum projection
    face = FaceDefinition()
    push_definition!(face, :NSPGaussLeg, 12)
    push_definition!(face, :NMPTrans, 3)
    push_definition!(face, :NSPStot, 2)
    push_definition!(face, :NOptTrans, 4)

    config = SimulationConfig(face)
    layout = ParameterLayout(2, 0, 0, 0)

    sim = ManyVariableVariationalMonteCarlo.EnhancedVMCSimulation{ComplexF64}(config, layout)
    sim.quantum_projection = initialize_quantum_projection_from_config(config; T=ComplexF64)

    info = ManyVariableVariationalMonteCarlo.get_quantum_projection_info(sim)

    @test info["enabled"] == true
    @test info["n_sp_gauss_leg"] == 12
    @test info["n_mp_trans"] == 3
    @test info["n_sp_stot"] == 2
    @test info["n_opt_trans"] == 4
    @test info["has_spin_projection"] == true  # NSPGaussLeg > 1
    @test info["has_momentum_projection"] == true  # NMPTrans > 1
    @test info["has_optimization_trans"] == true  # NOptTrans > 0

    # Test with disabled quantum projection
    sim_disabled = ManyVariableVariationalMonteCarlo.EnhancedVMCSimulation{ComplexF64}(config, layout)
    sim_disabled.quantum_projection = nothing

    info_disabled = ManyVariableVariationalMonteCarlo.get_quantum_projection_info(sim_disabled)

    @test info_disabled["enabled"] == false
    @test haskey(info_disabled, "reason")
end

@testitem "apply_quantum_projection_to_wavefunction! placeholder" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    # Create test simulation
    face = FaceDefinition()
    push_definition!(face, :L, 4)
    push_definition!(face, :model, "Spin")
    push_definition!(face, :NSPGaussLeg, 4)

    config = SimulationConfig(face)
    layout = ParameterLayout(2, 0, 0, 0)

    sim = ManyVariableVariationalMonteCarlo.EnhancedVMCSimulation{ComplexF64}(config, layout)
    sim.quantum_projection = initialize_quantum_projection_from_config(config; T=ComplexF64)

    # Test placeholder function
    electron_config = [1, 1, 0, 0]  # Two electrons on first two sites
    result = ManyVariableVariationalMonteCarlo.apply_quantum_projection_to_wavefunction!(sim, electron_config)

    # Currently returns unity as placeholder
    @test result == 1.0 + 0.0im

    # Test with no quantum projection
    sim.quantum_projection = nothing
    result_no_qp = ManyVariableVariationalMonteCarlo.apply_quantum_projection_to_wavefunction!(sim, electron_config)
    @test result_no_qp == 1.0 + 0.0im
end

@testitem "calculate_quantum_projection_ratio placeholder" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    # Create test simulation
    face = FaceDefinition()
    push_definition!(face, :L, 4)
    push_definition!(face, :model, "Spin")

    config = SimulationConfig(face)
    layout = ParameterLayout(2, 0, 0, 0)

    sim = ManyVariableVariationalMonteCarlo.EnhancedVMCSimulation{ComplexF64}(config, layout)
    sim.quantum_projection = initialize_quantum_projection_from_config(config; T=ComplexF64)

    # Test placeholder function
    old_config = [1, 1, 0, 0]
    new_config = [1, 0, 1, 0]

    ratio = ManyVariableVariationalMonteCarlo.calculate_quantum_projection_ratio(sim, old_config, new_config)

    # Currently returns unity as placeholder
    @test ratio == 1.0 + 0.0im
end

@testitem "quantum projection integration in enhanced simulation" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    fixture_path = joinpath(@__DIR__, "fixtures", "simple_spin_chain.def")

    if isfile(fixture_path)
        # Parse fixture file and add quantum projection parameters
        params = parse_stdface_def(fixture_path)

        # Create config with quantum projection
        config = ManyVariableVariationalMonteCarlo.stdface_to_simulation_config(params)

        # Manually add quantum projection parameters to face
        push_definition!(config.face, :NSPGaussLeg, 4)
        push_definition!(config.face, :NMPTrans, 1)

        layout = ManyVariableVariationalMonteCarlo.create_parameter_layout(config)

        sim = ManyVariableVariationalMonteCarlo.EnhancedVMCSimulation{ComplexF64}(config, layout)

        # Test initialization with quantum projection
        @test_nowarn ManyVariableVariationalMonteCarlo.initialize_enhanced_simulation!(sim)

        # Initialize quantum projection
        sim.quantum_projection = initialize_quantum_projection_from_config(config; T=ComplexF64)

        @test sim.quantum_projection !== nothing
        @test sim.quantum_projection.is_initialized == true

        # Test summary printing
        @test_nowarn print_quantum_projection_summary(sim)

    else
        @test_skip "Test fixture file not available"
    end
end

@testitem "quantum projection with different data types" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    # Test with Float64
    face = FaceDefinition()
    push_definition!(face, :NSPGaussLeg, 4)
    config = SimulationConfig(face)

    qp_f64 = initialize_quantum_projection_from_config(config; T=Float64)
    @test qp_f64 isa ManyVariableVariationalMonteCarlo.CCompatQuantumProjection{Float64}

    # Test with ComplexF64
    qp_cf64 = initialize_quantum_projection_from_config(config; T=ComplexF64)
    @test qp_cf64 isa ManyVariableVariationalMonteCarlo.CCompatQuantumProjection{ComplexF64}
end

@testitem "quantum projection parameter validation" begin
    using ManyVariableVariationalMonteCarlo
    using Test

    # Test with mismatched parameter array sizes
    face = FaceDefinition()
    push_definition!(face, :NMPTrans, 2)
    push_definition!(face, :ParaQPTrans, [1.0, 0.5, -0.5])  # Size mismatch: 3 vs 2

    config = SimulationConfig(face)

    # Should handle size mismatch gracefully
    qp = initialize_quantum_projection_from_config(config; T=ComplexF64)
    @test qp.n_mp_trans == 2
    # ParaQPTrans should not be set due to size mismatch
end
